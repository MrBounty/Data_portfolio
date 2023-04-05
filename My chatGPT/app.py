import sys
import openai
from PyQt5.QtGui import QTextCharFormat, QIcon, QFont, QColor
from PyQt5.QtWidgets import QComboBox, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QLineEdit, QPushButton, QListWidget, QListWidgetItem

# OpenAI variables
openai.api_key = "REPLACE_WITH_YOUR_API_KEY_HERE"
openai.api_base =  "REPLACE_WITH_YOUR_ENDPOINT_HERE"
openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'

# Define custom colors
bg_color = "#ffffff"
text_color = "#333333"
button_color = "#ffe600"
button_hover_color = "#cccccc"
input_border_color = "#cccccc"
input_bg_color = "#ffffff"
tab_bg_color = "#333333"
tab_text_color = "#ffffff"
tab_selected_bg_color = "#ffffff"
tab_selected_text_color = "#333333"
list_border_color = "#cccccc"
list_bg_color = "#ffffff"


# Define app style using custom colors
app_style = f"""
    QWidget {{
        background-color: {bg_color};
        color: {text_color};
    }}
    QPushButton {{
        background-color: {button_color};
        color: {text_color};
        padding: 8px;
        border: none;
        border-radius: 10px;
    }}
    QPushButton:hover {{
        background-color: {button_hover_color};
    }}
    QLineEdit {{
        padding: 8px;
        border: 1px solid {input_border_color};
        border-radius: 4px;
        background-color: {input_bg_color};
    }}
    QTextEdit {{
        padding: 8px;
        border: 1px solid {input_border_color};
        border-radius: 4px;
        background-color: {input_bg_color};
    }}
    QTabBar {{
        background-color: {tab_bg_color};
        color: {tab_text_color};
        height: 30px;
    }}
    QTabBar::tab {{
        background-color: {tab_bg_color};
        color: {tab_text_color};
        padding: 6px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }}
    QTabBar::tab:selected {{
        background-color: {tab_selected_bg_color};
        color: {tab_selected_text_color};
    }}
    QListWidget {{
        padding: 8px;
        border: 1px solid {list_border_color};
        border-radius: 4px;
        background-color: {list_bg_color};
    }}
"""

class ChatTab(QWidget):
    def __init__(self, model_combo):
        super().__init__()
        self.model_combo = model_combo
        self.first_message = True

        # Create widgets
        self.message_log = QTextEdit()
        self.message_log.setReadOnly(True)
        self.input = QLineEdit()
        self.button = QPushButton('Send')
        self.rename_input = QLineEdit()
        self.rename_button = QPushButton('Rename')
        self.rename_input.hide()
        self.rename_button.hide()

        # Set text format for message log
        fmt = QTextCharFormat()
        fmt.setFontFamily("Arial")
        fmt.setFontPointSize(12)
        fmt.setFontWeight(QFont.Normal)
        self.message_log.setCurrentCharFormat(fmt)

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.message_log)
        layout.addWidget(self.input)
        layout.addWidget(self.button)
        layout.addWidget(self.rename_input)
        layout.addWidget(self.rename_button)

        self.setLayout(layout)

        # Connect button clicks to functions
        self.button.clicked.connect(self.send_message)
        self.rename_button.clicked.connect(self.rename_chat_tab)

    def send_message(self):
        message = self.input.text()
        self.input.clear()

        # Display message in log
        self.message_log.insertHtml("<b>You:</b> " + message + "<br> <br>")
        
        # Get response from openai
        response = self.send_message_to_server(message)

        # Display response in log
        self.message_log.insertHtml("<b>Server:</b> " + response + "<br>")
        self.message_log.insertHtml("<hr style='color:#cccccc;'> <br>")

    def send_message_to_server(self, message):
        response = openai.ChatCompletion.create(
            engine = self.model_combo.currentData(),
            
            messages = 
            [
                {"role":"system","content":"You are an AI assistant."},
                {"role":"user","content":message}
            ] 
            )
        return response['choices'][0]['message']['content']

    def rename_chat_tab(self):
        new_name = self.rename_input.text()
        self.rename_input.clear()
        self.rename_input.hide()
        self.rename_button.hide()
        self.parent().setTabText(self.parent().currentIndex(), new_name)


class ChatBox(QWidget):
    def __init__(self):
        super().__init__()

        # Create widgets
        self.tab_widget = QTabWidget()
        self.chat_list_widget = QListWidget()
        self.model_combo = QComboBox()
        self.model_combo.addItem('GPT 3', 'gpt-35-turbo')
        self.model_combo.addItem('GPT 4', 'gpt-4')
        self.add_chat_button = QPushButton('New Chat')
        self.delete_chat_button = QPushButton('Delete Chat')
        self.rename_chat_button = QPushButton('Rename Chat')
        self.rename_input = QLineEdit()
        self.rename_button = QPushButton('Rename')
        self.rename_input.hide()
        self.rename_button.hide()

        # Create layout
        main_layout = QHBoxLayout()

        # Add chat list widget to left side of layout
        chat_list_layout = QVBoxLayout()
        chat_list_layout.addWidget(self.model_combo)
        chat_list_layout.addSpacing(5)
        chat_list_layout.addWidget(self.chat_list_widget)
        chat_list_layout.addSpacing(5)
        chat_list_layout.addWidget(self.add_chat_button)
        chat_list_layout.addSpacing(5)
        chat_list_layout.addWidget(self.delete_chat_button)
        chat_list_layout.addSpacing(5)
        chat_list_layout.addWidget(self.rename_chat_button)
        chat_list_layout.setContentsMargins(0, 0, 0, 0)
        chat_list_layout.setSpacing(0)
        chat_list_widget_container = QWidget()
        chat_list_widget_container.setLayout(chat_list_layout)
        chat_list_widget_container.setFixedWidth(150)
        main_layout.addWidget(chat_list_widget_container)

        # Add chat box to right side of layout
        chat_box_layout = QVBoxLayout()
        chat_box_layout.addWidget(self.tab_widget)
        chat_box_layout.addWidget(self.rename_input)
        chat_box_layout.addWidget(self.rename_button)
        chat_box_layout.setContentsMargins(0, 0, 0, 0)
        chat_box_layout.setSpacing(0)
        main_layout.addLayout(chat_box_layout)

        self.setLayout(main_layout)

        # Connect button clicks to functions
        self.add_chat_button.clicked.connect(self.add_chat_tab)
        self.delete_chat_button.clicked.connect(self.delete_chat_tab)
        self.rename_chat_button.clicked.connect(self.show_rename_input)
        self.rename_button.clicked.connect(self.rename_current_tab)

        # Connect chat list widget click to function
        self.chat_list_widget.itemClicked.connect(self.select_chat_tab)

        # Hide tab bar when there is only one tab
        self.tab_widget.setTabBarAutoHide(True)

        # Add initial chat tab
        self.add_chat_tab()

    def add_chat_tab(self):
        # Create new chat tab and add it to the tab widget
        chat_tab = ChatTab(self.model_combo)
        self.tab_widget.addTab(chat_tab, '')

        # Hide the tab bar
        self.tab_widget.tabBar().setVisible(False)

        # Add new chat to the chat list widget
        chat_item = QListWidgetItem(f"Chat {self.tab_widget.count()}")
        self.chat_list_widget.addItem(chat_item)

        # Select the new chat
        index = self.tab_widget.indexOf(chat_tab)
        self.tab_widget.setCurrentIndex(index)
        self.chat_list_widget.setCurrentRow(index)
  
    def delete_chat_tab(self):
        # Get currently selected chat tab and remove it from the tab widget
        current_index = self.tab_widget.currentIndex()
        self.tab_widget.removeTab(current_index)
        
        # Remove corresponding chat from the chat list widget
        chat_item = self.chat_list_widget.takeItem(current_index)
        del chat_item
        
        # Select the next chat in the chat list widget
        if self.chat_list_widget.count() > 0:
            next_index = (current_index + 1) % self.chat_list_widget.count()
            self.chat_list_widget.setCurrentRow(next_index)

        # Hide the tab bar
        self.tab_widget.tabBar().setVisible(False)
        self.chat_list_widget.setCurrentRow(current_index-1)
    
    def select_chat_tab(self, item):
        # Select the chat tab corresponding to the selected chat in the chat list widget
        index = self.chat_list_widget.row(item)
        self.tab_widget.setCurrentIndex(index)

    def show_rename_input(self):
        # Show the rename input and button
        self.rename_input.show()
        self.rename_button.show()
        
        # Set the text of the rename input to the current chat tab's name
        current_index = self.tab_widget.currentIndex()
        current_name = self.tab_widget.tabText(current_index)
        self.rename_input.setText(current_name)

    def rename_current_tab(self):
        # Get the text entered in the rename input
        new_name = self.rename_input.text()
        
        # Get the index of the current chat tab
        current_index = self.tab_widget.currentIndex()
        
        # Set the new name of the current chat tab
        self.tab_widget.setTabText(current_index, new_name)
        
        # Hide the rename input and button
        self.rename_input.hide()
        self.rename_button.hide()
        
        # Update the corresponding item in the chat list widget
        chat_item = self.chat_list_widget.item(current_index)
        chat_item.setText(new_name)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    app.setStyle('Fusion')
    app.setStyleSheet(app_style)
    
    # Create and show window
    window = ChatBox()
    window.setWindowTitle('My ChatGPT')
    window.setWindowIcon(QIcon('icon.png'))
    window.setGeometry(100, 100, 600, 400)
    window.show()
    
    sys.exit(app.exec_())
