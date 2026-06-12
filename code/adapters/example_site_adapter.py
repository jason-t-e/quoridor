from adapters.base_adapter import OnlinePlatformAdapter

# Note: You'll need to install playwright: `pip install playwright`
# In Colab: `!playwright install chromium`
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class ExampleSiteAdapter(OnlinePlatformAdapter):
    """
    Adapter for quoridor.sambaldwin.dev (or any future target site).

    Fallback / simulation mode:
      - `get_board_state()` returns None  →  caller uses local QuoridorGame state.
      - `is_my_turn()`      returns True  →  play_online handles alternation itself
                                             via `local_game.board.current_player`.
      - `is_game_over()`    returns False →  play_online terminates on local_game.is_terminal.
      - `make_move()`       is a no-op in fallback mode; caller applies move locally.

    To wire up a real website:
      1. Uncomment `self.page.goto(url)` in `connect()`.
      2. Implement the four TODO blocks below using Playwright selectors.
    """

    def __init__(self, settings):
        self.settings = settings
        self.website  = settings['online_play']['target_website']
        self.mode     = settings['online_play']['mode']
        self.playwright = None
        self.browser    = None
        self.context    = None
        self.page       = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self):
        if not sync_playwright:
            raise ImportError(
                "Playwright is not installed. "
                "Run: pip install playwright && playwright install chromium"
            )

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)

        record_video_dir = self.settings.get('online_play', {}).get('record_video_dir')
        if record_video_dir:
            import os
            os.makedirs(record_video_dir, exist_ok=True)
            self.context = self.browser.new_context(record_video_dir=record_video_dir)
            print(f"Video recording enabled → {record_video_dir}")
        else:
            self.context = self.browser.new_context()

        self.page = self.context.new_page()

        url = self.website if self.website.startswith("http") else f"https://{self.website}"
        print(f"Navigating to {url} in '{self.mode}' mode …")

        # ── TODO: uncomment when targeting a real site ──────────────────────
        # self.page.goto(url, wait_until="networkidle")
        # if self.mode == "account":
        #     self._login()

        print("(Fallback / simulation mode — not loading real URL)")

    def _login(self):
        creds = self.settings['online_play']['credentials']
        print(f"Logging in as: {creds['username']}")
        # TODO: fill real CSS selectors
        # self.page.fill('#username', creds['username'])
        # self.page.fill('#password', creds['password'])
        # self.page.click('#login-button')
        # self.page.wait_for_selector('#game-board', timeout=10_000)

    # ── Game interface ────────────────────────────────────────────────────────

    def get_board_state(self):
        """
        Parse the live board from the website's DOM and return a BoardState.
        Returns None in fallback/simulation mode — the caller will use the
        local QuoridorGame state instead.
        """
        # TODO: implement real DOM parsing, e.g.:
        # cells = self.page.query_selector_all('.board-cell')
        # ... build and return a BoardState object ...
        return None   # fallback

    def make_move(self, action):
        """
        Translate an engine Move into a click / key event on the website.
        In fallback mode this is a no-op; the caller applies the move locally.
        """
        # TODO: implement real click logic, e.g.:
        # from engine.moves import PawnMove, WallMove
        # if isinstance(action, PawnMove):
        #     self.page.click(f'#cell-{action.to_row}-{action.to_col}')
        # elif isinstance(action, WallMove):
        #     self.page.click(f'#wall-{action.row}-{action.col}-{action.orientation}')
        pass   # fallback: no-op

    def is_my_turn(self) -> bool:
        """
        Returns True if it is our bot's turn on the website.
        In fallback mode always returns True; play_online uses local game state
        to determine actual alternation.
        """
        # TODO: implement real turn-indicator check, e.g.:
        # return self.page.is_visible('#your-turn-banner')
        return True   # fallback

    def is_game_over(self) -> bool:
        """
        Returns True if the game has finished on the website.
        In fallback mode always returns False; play_online terminates on
        local_game.is_terminal instead.
        """
        # TODO: implement real game-over detection, e.g.:
        # return self.page.is_visible('#game-over-modal')
        return False   # fallback

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        try:
            if self.context:
                self.context.close()
        except Exception:
            pass
        try:
            if self.browser:
                self.browser.close()
        except Exception:
            pass
        try:
            if self.playwright:
                self.playwright.stop()
        except Exception:
            pass
