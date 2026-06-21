import os
import json
import logging
from typing import Optional
from playwright.sync_api import sync_playwright, Page, BrowserContext
from game_engine.moves import Move, PawnMove, WallMove
from game_engine.board import BoardState

class PlaywrightAdapter:
    def __init__(self, headless: bool = True, record_dir: str = "data/recordings/"):
        self.headless = headless
        self.record_dir = record_dir
        self.playwright = None
        self.browser = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        os.makedirs(self.record_dir, exist_ok=True)
        
    def start(self, url: str = "https://quoridor.sambaldwin.dev/"):
        self.playwright = sync_playwright().start()
        # Launch browser
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        # Create context with video recording
        self.context = self.browser.new_context(record_video_dir=self.record_dir)
        self.page = self.context.new_page()
        
        try:
            self.page.goto(url)
            logging.info(f"Navigated to {url}")
        except Exception as e:
            logging.error(f"Failed to navigate: {e}")
            self.close()
            raise e

    def parse_board_state(self) -> BoardState:
        """
        Dummy implementation. In a real scenario, this would use self.page.evaluate()
        or element locators to parse the HTML and return a BoardState.
        """
        # Example of how we might read a board state from DOM:
        # dom_state = self.page.evaluate("() => getBoardStateFromDOM()")
        return BoardState()

    def execute_move(self, move: Move):
        """
        Translate the internal Move object to a click on the actual DOM.
        """
        if isinstance(move, PawnMove):
            # Example: self.page.locator(f"#cell-{move.to_row}-{move.to_col}").click()
            pass
        elif isinstance(move, WallMove):
            # Example: self.page.locator(f"#wall-{move.row}-{move.col}-{move.orientation}").click()
            pass

    def check_game_over(self) -> bool:
        """
        Check if the DOM indicates the game has ended.
        """
        return False

    def close(self):
        """
        CRITICAL: Ensure the page and context are closed so .webm videos flush to disk.
        """
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            logging.info("Playwright session closed securely. Videos flushed.")
        except Exception as e:
            logging.error(f"Error during playwright shutdown: {e}")
