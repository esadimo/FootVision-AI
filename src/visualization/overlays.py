"""
src.visualization.overlays — Overlay drawing and live visualization window managers.
"""

import cv2
import numpy as np

def show_frame(window_name: str, frame: np.ndarray, delay_ms: int = 0) -> bool:
    """
    Displays a frame in an interactive OpenCV pop-up window.
    
    Parameters
    ----------
    window_name : str
        The title of the display window.
    frame : np.ndarray
        The BGR image array to display.
    delay_ms : int
        Milliseconds to wait for keypress. 
        0 means wait indefinitely (perfect for single images).
        1 or higher means wait that long before continuing (perfect for video loops).

    Returns
    -------
    bool
        True if processing should continue, False if the user requested to Quit ('q' or ESC).
    """
    # Create a resizable window to fit different monitor scales
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Scale window bounds to a reasonable starting scale while preserving aspect ratio
    h, w = frame.shape[:2]
    max_w, max_h = 1280, 720
    scale = min(max_w / w, max_h / h)
    if scale < 1.0:
        cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
    else:
        cv2.resizeWindow(window_name, w, h)
        
    cv2.imshow(window_name, frame)
    
    # Listen to keyboard input
    key = cv2.waitKey(delay_ms) & 0xFF
    
    # Check for quit command: 'q' key (113) or ESC (27)
    if key == ord('q') or key == 27:
        print("\n  [Viewer] Quit request received. Closing window...")
        cv2.destroyWindow(window_name)
        return False
        
    # Check for pause toggle: Spacebar (32)
    elif key == ord(' '):
        print("\n  [Viewer] Paused. Press Spacebar to resume, or 'q' to quit.")
        while True:
            paused_key = cv2.waitKey(100) & 0xFF
            if paused_key == ord(' '):
                print("  [Viewer] Resuming...")
                break
            if paused_key == ord('q') or paused_key == 27:
                print("  [Viewer] Quit request received during pause. Closing window...")
                cv2.destroyWindow(window_name)
                return False
                
    return True

def close_all_windows() -> None:
    """Closes all active OpenCV display windows."""
    cv2.destroyAllWindows()
