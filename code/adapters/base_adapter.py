from abc import ABC, abstractmethod

class OnlinePlatformAdapter(ABC):
    """
    Abstract base class for interacting with online Quoridor platforms.
    Different websites will have different implementations of this class.
    """

    @abstractmethod
    def connect(self):
        """
        Connect to the platform. This might involve opening a browser,
        navigating to the URL, and logging in or entering guest mode.
        """
        pass

    @abstractmethod
    def get_board_state(self):
        """
        Extract the current board state from the website.
        Returns a representation of the board that the Quoridor engine can understand.
        """
        pass

    @abstractmethod
    def make_move(self, action):
        """
        Execute a move on the website.
        'action' is the internal representation of the move (e.g., a coordinate or wall placement).
        """
        pass

    @abstractmethod
    def is_my_turn(self):
        """
        Check if it's the bot's turn to play.
        """
        pass

    @abstractmethod
    def is_game_over(self):
        """
        Check if the current game has finished.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the connection / browser.
        """
        pass
