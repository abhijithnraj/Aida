import sys
import os
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTextEdit, QPushButton, QLineEdit, 
                           QLabel, QScrollArea, QDialog, QMessageBox,QSizePolicy,
                           QFrame, QTextBrowser, QInputDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QTextCursor, QPalette, QColor, QFont, QIcon
from .core import Aida
from .config import AidaConfig

class LoadingDots(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("color: white;")
        self.dots = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_dots)
        self.setText("Processing")
    
    def start(self):
        self.timer.start(500)
    
    def stop(self):
        self.timer.stop()
        self.hide()
    
    def update_dots(self):
        self.dots = (self.dots + 1) % 4
        self.setText("Processing" + "." * self.dots)

    executed = pyqtSignal(str)
class CommandBubble(QFrame):
    rejected = pyqtSignal(str)
    executed = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # New signal for command execution result
    
    def __init__(self, command, parent=None):
        super().__init__(parent)
        self.command = command
        self.setup_ui()
        self.is_finished = False
        self.worker = None
    
    def setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        
        # Set color scheme
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(52, 53, 65))
        pal.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.setPalette(pal)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Command display
        command_layout = QHBoxLayout()
        command_label = QLabel("Command:")
        command_label.setStyleSheet("color: #4a9eff;")
        self.command_text = QLabel(self.command)
        self.command_text.setStyleSheet("color: white;")
        self.command_text.setWordWrap(True)
        command_layout.addWidget(command_label)
        command_layout.addWidget(self.command_text, 1)
        layout.addLayout(command_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.modify_btn = QPushButton("Modify")
        self.modify_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #4a9eff;
                border: 1px solid #4a9eff;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: rgba(74, 158, 255, 0.1);
            }
        """)
        
        self.accept_btn = QPushButton("✓")
        self.accept_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        self.reject_btn = QPushButton("✗")
        self.reject_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        button_layout.addWidget(self.modify_btn)
        button_layout.addWidget(self.accept_btn)
        button_layout.addWidget(self.reject_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Loading indicator (hidden by default)
        self.loading = LoadingDots(self)
        self.loading.hide()
        layout.addWidget(self.loading)
        
        # Connect signals
        self.modify_btn.clicked.connect(self.modify_command)
        self.accept_btn.clicked.connect(self.accept_command)
        self.reject_btn.clicked.connect(self.reject_command)
        
        # Set size policy to fit content
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.adjustSize()
    
    def modify_command(self):
        if self.is_finished:
            return
            
        new_command, ok = QInputDialog.getText(
            self, "Modify Command", "Edit command:", 
            QLineEdit.EchoMode.Normal, self.command
        )
        if ok and new_command:
            self.command = new_command
            self.command_text.setText(new_command)
    
    def accept_command(self):
        if self.is_finished:
            return
            
        self.loading.show()
        self.loading.start()
        self.modify_btn.setEnabled(False)
        self.accept_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
        
        # Check if it's a sudo command
        if self.command.strip().startswith("sudo "):
            password, ok = QInputDialog.getText(
                self, "Sudo Password", "Enter sudo password:", 
                QLineEdit.EchoMode.Password
            )
            if ok:
                self.worker = CommandExecutionWorker(f"SUDO_PASSWORD={password}\n{self.command}")
            else:
                self.loading.stop()
                if not self.is_finished:
                    self.modify_btn.setEnabled(True)
                    self.accept_btn.setEnabled(True)
                    self.reject_btn.setEnabled(True)
                return
        else:
            self.worker = CommandExecutionWorker(self.command)
        
        if self.worker:
            self.worker.finished.connect(self._handle_execution_result)
            self.worker.start()
    
    def _handle_execution_result(self, success, result):
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        
        self.finished.emit(success, result)
        self.command_finished()
    
    def command_finished(self):
        if not self.is_finished:
            self.is_finished = True
            self.loading.stop()
            QTimer.singleShot(100, self.safe_cleanup)
    
    def safe_cleanup(self):
        if not self.is_finished:
            self.is_finished = True
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        self.hide()
        self.deleteLater()
    
    def reject_command(self):
        if self.is_finished:
            return
            
        feedback, ok = QInputDialog.getText(
            self, "Feedback", "Please provide feedback for the AI:",
            QLineEdit.EchoMode.Normal
        )
        if ok:
            self.rejected.emit(feedback)
            self.safe_cleanup()

class MessageBubble(QFrame):
    """Custom widget for chat message bubbles"""
    def __init__(self, text, is_user=False, has_thought_process=False, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        
        # Set color scheme
        pal = self.palette()
        if is_user:
            pal.setColor(QPalette.ColorRole.Window, QColor(0, 132, 255))
            pal.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        else:
            pal.setColor(QPalette.ColorRole.Window, QColor(52, 53, 65))
            pal.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.setPalette(pal)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Message content
        self.content = QTextBrowser()
        self.content.setOpenExternalLinks(True)
        self.content.setStyleSheet("""
            QTextBrowser {
                background-color: transparent;
                border: none;
                color: white;
            }
        """)
        self.content.setFont(QFont("Segoe UI", 10))
        self.content.setHtml(text)
        
        # Set size policy to fit content
        from PyQt6 import QtWidgets
        self.content.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        layout.addWidget(self.content)
        
        # Loading indicator (hidden by default)
        self.loading = LoadingDots()
        self.loading.hide()
        layout.addWidget(self.loading)
        
        # Thought process (for AI messages only)
        if has_thought_process:
            self.thought_button = QPushButton("Show thought process")
            self.thought_button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #4a9eff;
                    border: 1px solid #4a9eff;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: rgba(74, 158, 255, 0.1);
                }
            """)
            self.thought_button.clicked.connect(self.toggle_thought_process)
            layout.addWidget(self.thought_button)
            
            self.thought_content = QTextBrowser()
            self.thought_content.setStyleSheet("""
                QTextBrowser {
                    background-color: rgba(0, 0, 0, 0.2);
                    border: none;
                    color: #ccc;
                    margin: 5px;
                    padding: 5px;
                    border-radius: 3px;
                }
            """)
            self.thought_content.hide()
            layout.addWidget(self.thought_content)
        
        # Set size policy to fit content
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.adjustSize()
    
    def start_loading(self):
        self.loading.show()
        self.loading.start()
    
    def stop_loading(self):
        self.loading.stop()
    
    def set_thought_process(self, text):
        if hasattr(self, 'thought_content'):
            self.thought_content.setPlainText(text)
    
    def toggle_thought_process(self):
        if hasattr(self, 'thought_content'):
            if self.thought_content.isHidden():
                self.thought_content.show()
                self.thought_button.setText("Hide thought process")
            else:
                self.thought_content.hide()
                self.thought_button.setText("Show thought process")
            self.adjustSize()

class ChatWidget(QWidget):
    """Widget to display chat messages"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        
        # Chat display area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.messages_layout.addStretch()
        
        self.scroll_area.setWidget(self.messages_widget)
        layout.addWidget(self.scroll_area)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(100)
        self.input_field.setFont(QFont("Segoe UI", 10))
        
        self.send_button = QPushButton("Send")
        self.send_button.setFixedWidth(100)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        self.setLayout(layout)
    
    def add_message(self, text, is_user=False, thought_process=None):
        # Remove the stretch if it exists
        if self.messages_layout.count() > 0:
            stretch_item = self.messages_layout.itemAt(self.messages_layout.count() - 1)
            if stretch_item.spacerItem():
                self.messages_layout.removeItem(stretch_item)
        
        # Create message bubble
        bubble = MessageBubble(text, is_user, has_thought_process=bool(thought_process))
        if thought_process:
            bubble.set_thought_process(thought_process)
        
        # Create container for alignment
        container = QHBoxLayout()
        container.addStretch() if is_user else None
        container.addWidget(bubble, alignment=Qt.AlignmentFlag.AlignVCenter)
        bubble.setMaximumWidth(int(self.width() * 0.7))
        container.addStretch() if not is_user else None
        
        self.messages_layout.addLayout(container)
        self.messages_layout.addStretch()
        
        # Scroll to bottom with animation
        QTimer.singleShot(100, self.scroll_to_bottom)
        
        return bubble
    
    def scroll_to_bottom(self):
        vsb = self.scroll_area.verticalScrollBar()
        vsb.setValue(vsb.maximum())
    
    def get_input(self):
        return self.input_field.toPlainText().strip()
    
    def clear_input(self):
        self.input_field.clear()

from langchain_community.tools import ShellTool

class CommandExecutionWorker(QThread):
    finished = pyqtSignal(bool, str)
    
    def __init__(self, command):
        super().__init__()
        self.command = command
        self.shell_tool = ShellTool()
    
    def run(self):
        try:
            # Check if it's a sudo command
            if self.command.strip().startswith("SUDO_PASSWORD="):
                parts = self.command.split("\n", 1)
                if len(parts) == 2:
                    sudo_password = parts[0].split("=", 1)[1]
                    command = parts[1]
                    result = self.shell_tool.run(f"echo {sudo_password} | sudo -S {command}")
                else:
                    result = "Invalid sudo command format"
                    self.finished.emit(False, result)
                    return
            else:
                result = self.shell_tool.run(self.command)
            self.finished.emit(True, result)
        except Exception as e:
            self.finished.emit(False, str(e))

class ApiKeyDialog(QDialog):
    """Dialog for entering Gemini API key"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gemini API Key")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Add description
        description = QLabel("Enter your Gemini API key. You can get one from:\nhttps://makersuite.google.com/app/apikey")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Add input field
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.key_input)
        
        # Add buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_api_key(self):
        return self.key_input.text().strip()

class AidaWorker(QThread):
    """Worker thread to handle AIDA processing"""
    finished = pyqtSignal(str, str)  # Response, Thought process
    
    def __init__(self, aida, query):
        super().__init__()
        self.aida = aida
        self.query = query
    
    def run(self):
        try:
            response = self.aida.process_query(self.query)
            
            # Extract thought process if available
            thought_process = ""
            if "Thought:" in response:
                parts = response.split("Final Answer:", 1)
                if len(parts) == 2:
                    thought_process = parts[0].strip()
                    response = parts[1].strip()
            
            self.finished.emit(response, thought_process)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}", "")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIDA - AI Server Management Assistant")
        self.setMinimumSize(800, 600)
        
        # Initialize AIDA
        self.initialize_aida()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Add API key button
        api_key_layout = QHBoxLayout()
        api_key_layout.addStretch()
        self.api_key_button = QPushButton("Set Gemini API Key")
        self.api_key_button.clicked.connect(self.show_api_key_dialog)
        api_key_layout.addWidget(self.api_key_button)
        layout.addLayout(api_key_layout)
        
        # Add chat widget
        self.chat_widget = ChatWidget()
        self.chat_widget.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.chat_widget)
        
        # Welcome message
        self.chat_widget.add_message(
            "Welcome! I'm AIDA, your AI server management assistant. How can I help you today?")
    
    def validate_command(self, command, callback):
        """GUI-based command validation using CommandBubble"""
        bubble = CommandBubble(command, parent=self.chat_widget.messages_widget)
        container = QHBoxLayout()
        container.addStretch()
        container.addWidget(bubble, alignment=Qt.AlignmentFlag.AlignVCenter)
        bubble.setMaximumWidth(int(self.width() * 0.7))
        container.addStretch()
        
        self.chat_widget.messages_layout.addLayout(container)
        self.chat_widget.scroll_to_bottom()
        
        def handle_execution_result(success, result):
            if success:
                self.chat_widget.add_message(f"Command output:\n{result}")
            else:
                self.chat_widget.add_message(f"Command failed:\n{result}")
            callback(success, result)
        
        def handle_rejection(feedback):
            self.chat_widget.add_message(f"Command rejected. Feedback: {feedback}")
            callback(False, "Command rejected by user")
        
        bubble.finished.connect(handle_execution_result)
        bubble.rejected.connect(handle_rejection)
    
    def initialize_aida(self):
        """Initialize AIDA with configuration"""
        try:
            config = AidaConfig()
            self.aida = Aida(config=config, gui_validator=self.validate_command)
            print("AIDA initialized")
        except ValueError as e:
            # Don't show error message for missing API key
            if "GOOGLE_API_KEY" not in str(e):
                QMessageBox.critical(self, "Error", 
                                   f"Failed to initialize AIDA: {str(e)}")
    
    def show_api_key_dialog(self):
        """Show dialog to enter Gemini API key"""
        dialog = ApiKeyDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            api_key = dialog.get_api_key()
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                self.initialize_aida()
                QMessageBox.information(self, "Success", 
                                      "API key saved successfully!")
    
    def send_message(self):
        """Handle sending a message"""
        message = self.chat_widget.get_input()
        if not message:
            return
        
        if not hasattr(self, 'aida'):
            try:
                self.initialize_aida()
            except ValueError as e:
                QMessageBox.warning(self, "API Key Required", 
                                  "Please set your Gemini API key first.")
                return
        
        # Add user message to chat
        self.chat_widget.add_message(message, is_user=True)
        self.chat_widget.clear_input()
        
        # Add AI message bubble with loading animation
        ai_bubble = self.chat_widget.add_message("", thought_process="")
        ai_bubble.start_loading()
        
        # Disable input while processing
        self.chat_widget.input_field.setEnabled(False)
        self.chat_widget.send_button.setEnabled(False)
        
        # Process in background
        self.worker = AidaWorker(self.aida, message)
        self.worker.finished.connect(lambda response, thought: self.handle_response(ai_bubble, response, thought))
        self.worker.start()
    
    def handle_response(self, bubble, response, thought_process):
        """Handle AIDA's response"""
        bubble.stop_loading()
        bubble.content.setHtml(response)
        if thought_process:
            bubble.set_thought_process(thought_process)
        
        # Re-enable input
        self.chat_widget.input_field.setEnabled(True)
        self.chat_widget.send_button.setEnabled(True)
        self.chat_widget.input_field.setFocus()

def main():
    app = QApplication(sys.argv)
    
    # Set style
    app.setStyle("Fusion")
    
    # Set dark theme
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(32, 33, 35))  # Darker background
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
