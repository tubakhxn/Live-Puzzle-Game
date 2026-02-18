import cv2
import numpy as np
import random

class PuzzleUI:
    def __init__(self, img, grid_size=3):
        self.img = img
        self.grid_size = grid_size
        self.pieces, self.ph, self.pw = self.split_image(img, grid_size)
        self.shuffled_pieces, self.order = self.shuffle_pieces(self.pieces)
        self.selected = None
        self.window = np.zeros_like(img)
        self.positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        self.current_order = self.order.copy()
        self.dragging = False
        self.drag_idx = None
        self.offset = (0, 0)
        self.gesture_dragging = False
        self.gesture_drag_idx = None
        self.gesture_drag_pos = None
        self.gesture_last_drop_idx = None
        cv2.namedWindow("Live Puzzle")
        # Mouse callback is not used for gesture mode

    def get_piece_at(self, x, y):
        col = x // self.pw
        row = y // self.ph
        idx = row * self.grid_size + col
        if 0 <= idx < len(self.shuffled_pieces):
            return idx
        return None

    def gesture_pick(self, x, y):
        idx = self.get_piece_at(x, y)
        if idx is not None:
            self.gesture_dragging = True
            self.gesture_drag_idx = idx
            self.gesture_drag_pos = (x, y)
            return True
        return False

    def gesture_drag(self, x, y):
        if self.gesture_dragging:
            self.gesture_drag_pos = (x, y)

    def gesture_drop(self, x, y):
        if self.gesture_dragging:
            drop_idx = self.get_piece_at(x, y)
            if drop_idx is not None and drop_idx != self.gesture_drag_idx:
                # Swap pieces
                self.shuffled_pieces[self.gesture_drag_idx], self.shuffled_pieces[drop_idx] = self.shuffled_pieces[drop_idx], self.shuffled_pieces[self.gesture_drag_idx]
                self.current_order[self.gesture_drag_idx], self.current_order[drop_idx] = self.current_order[drop_idx], self.current_order[self.gesture_drag_idx]
                self.gesture_last_drop_idx = drop_idx
            self.gesture_dragging = False
            self.gesture_drag_idx = None
            self.gesture_drag_pos = None


    def split_image(self, img, grid_size):
        h, w = img.shape[:2]
        ph, pw = h // grid_size, w // grid_size
        pieces = []
        for i in range(grid_size):
            for j in range(grid_size):
                piece = img[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                pieces.append(piece)
        return pieces, ph, pw

    def shuffle_pieces(self, pieces):
        idx = list(range(len(pieces)))
        random.shuffle(idx)
        return [pieces[i] for i in idx], idx

    def mouse_event(self, event, x, y, flags, param):
        col = x // self.pw
        row = y // self.ph
        idx = row * self.grid_size + col
        if event == cv2.EVENT_LBUTTONDOWN:
            if idx < len(self.shuffled_pieces):
                self.dragging = True
                self.drag_idx = idx
                self.offset = (x - (col * self.pw), y - (row * self.ph))
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                drop_col = x // self.pw
                drop_row = y // self.ph
                drop_idx = drop_row * self.grid_size + drop_col
                if drop_idx < len(self.shuffled_pieces):
                    # Swap pieces
                    self.shuffled_pieces[self.drag_idx], self.shuffled_pieces[drop_idx] = self.shuffled_pieces[drop_idx], self.shuffled_pieces[self.drag_idx]
                    self.current_order[self.drag_idx], self.current_order[drop_idx] = self.current_order[drop_idx], self.current_order[self.drag_idx]
                self.dragging = False
                self.drag_idx = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.mouse_x, self.mouse_y = x, y

    def show(self, dark_theme=False, window_name="Puzzle"):
        if dark_theme:
            self.window = np.zeros_like(self.img)
        else:
            self.window = np.zeros_like(self.img) + 255
        border = 2
        # Draw all pieces with a border for clarity
        for idx, piece in enumerate(self.shuffled_pieces):
            row = idx // self.grid_size
            col = idx % self.grid_size
            y, x = row * self.ph, col * self.pw
            if self.gesture_dragging and idx == self.gesture_drag_idx:
                continue  # Don't draw the dragged piece here
            self.window[y:y+self.ph, x:x+self.pw] = piece
            cv2.rectangle(self.window, (x, y), (x+self.pw, y+self.ph), (200,200,200), border)
        # Draw dragged piece at gesture position with strong highlight (mouse drag not gesture)
        if self.dragging and self.drag_idx is not None:
            x, y = getattr(self, 'mouse_x', 0), getattr(self, 'mouse_y', 0)
            piece = self.shuffled_pieces[self.drag_idx]
            y0 = y - self.offset[1]
            x0 = x - self.offset[0]
            y1 = y0 + self.ph
            x1 = x0 + self.pw
            # Clip to window
            y0 = max(0, y0)
            x0 = max(0, x0)
            y1 = min(self.window.shape[0], y1)
            x1 = min(self.window.shape[1], x1)
            piece_crop = piece[:y1-y0, :x1-x0]
            self.window[y0:y1, x0:x1] = piece_crop
            cv2.rectangle(self.window, (x0, y0), (x1, y1), (0,255,255), 4)
        cv2.imshow(window_name, self.window)
        if self.current_order == list(range(len(self.pieces))):
            return True
        return False
