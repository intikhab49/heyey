from typing import Dict, Set, Optional, List
import asyncio
import logging
from datetime import datetime, timedelta
import pytz
from fastapi import WebSocket
from .prediction import predict_next_price, ensemble_prediction
from simple_config import settings

logger = logging.getLogger(__name__)

class SymbolManager:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.connections: Dict[str, Set[WebSocket]] = {}  # timeframe -> connections
        self.last_data: Dict[str, dict] = {}  # timeframe -> last data
        self.update_tasks: Dict[str, asyncio.Task] = {}  # timeframe -> task
        self.error_counts: Dict[str, int] = {}  # timeframe -> error count

    def add_connection(self, websocket: WebSocket, timeframe: str):
        if timeframe not in self.connections:
            self.connections[timeframe] = set()
        self.connections[timeframe].add(websocket)

    def remove_connection(self, websocket: WebSocket, timeframe: str):
        if timeframe in self.connections:
            self.connections[timeframe].discard(websocket)
            if not self.connections[timeframe]:
                del self.connections[timeframe]
                if timeframe in self.update_tasks:
                    self.update_tasks[timeframe].cancel()
                    del self.update_tasks[timeframe]

    def has_connections(self, timeframe: str) -> bool:
        return bool(self.connections.get(timeframe, set()))

    def get_last_data(self, timeframe: str) -> Optional[dict]:
        return self.last_data.get(timeframe)

    def set_last_data(self, timeframe: str, data: dict):
        self.last_data[timeframe] = data

    def increment_error(self, timeframe: str) -> int:
        self.error_counts[timeframe] = self.error_counts.get(timeframe, 0) + 1
        return self.error_counts[timeframe]

    def reset_error(self, timeframe: str):
        self.error_counts[timeframe] = 0

class WebSocketManager:
    def __init__(self):
        self.symbols: Dict[str, SymbolManager] = {}
        self.update_intervals = settings.WS_UPDATE_INTERVAL
        self.max_errors = settings.WS_MAX_ERRORS

    def get_symbol_manager(self, symbol: str) -> SymbolManager:
        if symbol not in self.symbols:
            self.symbols[symbol] = SymbolManager(symbol)
        return self.symbols[symbol]

    async def connect(self, websocket: WebSocket, symbol: str, timeframe: str = "1h"):
        """Connect a client to receive updates for a symbol and timeframe"""
        await websocket.accept()
        
        symbol_manager = self.get_symbol_manager(symbol)
        symbol_manager.add_connection(websocket, timeframe)

        # Send last known data immediately if available
        last_data = symbol_manager.get_last_data(timeframe)
        if last_data:
            try:
                await websocket.send_json(last_data)
            except Exception as e:
                logger.error(f"Failed to send initial data: {str(e)}")

        # Start update task if not already running
        if timeframe not in symbol_manager.update_tasks:
            symbol_manager.update_tasks[timeframe] = asyncio.create_task(
                self._update_loop(symbol, timeframe)
            )

        logger.info(f"Client connected for {symbol} ({timeframe})")

    def disconnect(self, websocket: WebSocket, symbol: str, timeframe: str):
        """Disconnect a client and cleanup if needed"""
        try:
            symbol_manager = self.symbols.get(symbol)
            if symbol_manager:
                symbol_manager.remove_connection(websocket, timeframe)
                if not symbol_manager.connections:
                    del self.symbols[symbol]
        except Exception as e:
            logger.error(f"Error in disconnect: {str(e)}")

    async def broadcast_to_symbol(self, symbol: str, timeframe: str, message: dict):
        """Broadcast message to all clients watching a symbol/timeframe"""
        try:
            symbol_manager = self.symbols.get(symbol)
            if not symbol_manager:
                return

            dead_connections = set()
            for connection in symbol_manager.connections.get(timeframe, set()):
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to client: {str(e)}")
                    dead_connections.add(connection)

            # Cleanup dead connections
            for dead in dead_connections:
                self.disconnect(dead, symbol, timeframe)

        except Exception as e:
            logger.error(f"Error in broadcast: {str(e)}")

    async def _update_loop(self, symbol: str, timeframe: str):
        """Background task to update data for a symbol/timeframe"""
        symbol_manager = self.symbols[symbol]
        interval = self.update_intervals.get(timeframe, 3600)
        last_update_time = datetime.now(pytz.UTC)

        while symbol_manager.has_connections(timeframe):
            try:
                current_time = datetime.now(pytz.UTC)
                time_since_update = (current_time - last_update_time).total_seconds()

                # Only update if enough time has passed
                if time_since_update < interval:
                    await asyncio.sleep(1)
                    continue

                # Get prediction based on timeframe
                if timeframe == "24h":
                    data = ensemble_prediction(symbol)
                else:
                    data = predict_next_price(symbol, timeframe)

                if "error" not in data:
                    message = {
                        "type": "update",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": current_time.isoformat(),
                        "data": data,
                        "next_update": (current_time + timedelta(seconds=interval)).isoformat()
                    }

                    # Store last successful data
                    symbol_manager.set_last_data(timeframe, message)
                    # Reset error count on success
                    symbol_manager.reset_error(timeframe)
                    last_update_time = current_time

                    await self.broadcast_to_symbol(symbol, timeframe, message)
                    logger.debug(f"Update sent for {symbol} ({timeframe})")
                else:
                    error_count = symbol_manager.increment_error(timeframe)
                    error_message = {
                        "type": "error",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": current_time.isoformat(),
                        "error": data["error"]
                    }
                    await self.broadcast_to_symbol(symbol, timeframe, error_message)
                    logger.error(f"Prediction error for {symbol} ({timeframe}): {data['error']}")
                    
                    if error_count >= self.max_errors:
                        logger.error(f"Too many errors for {symbol} ({timeframe}), stopping updates")
                        break

            except Exception as e:
                error_count = symbol_manager.increment_error(timeframe)
                logger.error(f"Error updating {symbol} ({timeframe}): {str(e)}")
                
                try:
                    error_message = {
                        "type": "error",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": datetime.now(pytz.UTC).isoformat(),
                        "error": str(e)
                    }
                    await self.broadcast_to_symbol(symbol, timeframe, error_message)
                except Exception as broadcast_error:
                    logger.error(f"Failed to broadcast error message: {str(broadcast_error)}")
                
                if error_count >= self.max_errors:
                    logger.error(f"Too many errors for {symbol} ({timeframe}), stopping updates")
                    break

                await asyncio.sleep(min(interval, 60))  # Wait at most 1 minute on error
            
            await asyncio.sleep(1)  # Small sleep to prevent CPU overuse

    async def handle_client_message(self, websocket: WebSocket, symbol: str, timeframe: str, message: dict):
        """Handle incoming messages from clients"""
        try:
            msg_type = message.get("type")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif msg_type == "subscribe":
                new_timeframe = message.get("timeframe", timeframe)
                if new_timeframe != timeframe:
                    self.disconnect(websocket, symbol, timeframe)
                    await self.connect(websocket, symbol, new_timeframe)
            
            elif msg_type == "unsubscribe":
                self.disconnect(websocket, symbol, timeframe)
            
            elif msg_type == "request_update":
                symbol_manager = self.symbols.get(symbol)
                if symbol_manager:
                    last_data = symbol_manager.get_last_data(timeframe)
                    if last_data:
                        await websocket.send_json(last_data)

        except Exception as e:
            logger.error(f"Error handling client message: {str(e)}")

manager = WebSocketManager() 