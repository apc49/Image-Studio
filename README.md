# AWDigitalworld Image Analyzer

AWDigitalworld Image Analyzer is a modular, AI-powered image analysis tool designed to enhance productivity in handling images and managing datasets. This app integrates with machine learning models and other AI features for tasks like image recognition, facial detection, and more.

## Features

- **Image Gallery**: Manage and view images with the ability to add captions, tags, and other metadata.
- **Image Analysis**: Analyze images with built-in machine learning models and AI-powered tools.
- **AI Integration**: Integrates GPT and DALL·E for enhanced image generation and text summarization.
- **Database Management**: Manage images, prompts, and galleries using SQLite.
- **Interactive UI**: A modern, responsive GUI built with Tkinter for an intuitive experience.
- **System Monitoring**: Real-time CPU and RAM usage monitoring.
- **Customizable Settings**: Toggle between light and dark themes, enable or disable certain AI features, and adjust settings based on user needs.
- **Security**: Built-in security features for local area network (LAN) protection.

## Installation

### Prerequisites

- Python 3.10+
- Required Python libraries (can be installed via `requirements.txt`)

### Steps

1. **Clone or Download the Repo**
   - Clone the repository:
     ```bash
     git clone https://github.com/apc49/Image-Studio.git
     ```
   - Or download the ZIP file directly from GitHub.

2. **Install Dependencies**
   - Ensure you have Python 3.10+ installed.
   - Install the necessary dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Running the Application**
   - After dependencies are installed, run the main application:
     ```bash
     python main.py
     ```

4. **Optional: Create a Virtual Environment**
   - If you want to use a virtual environment, follow these steps:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
     pip install -r requirements.txt
     python main.py
     ```

## Usage

1. **Start the Application**
   - When you first run the app, you'll be greeted by a splash screen that gives you an overview of the features and upcoming updates.

2. **Main Features**
   - **Gallery Interface**: View, add, and remove images from the gallery.
   - **Image Analysis**: Click on any image in the gallery to analyze it. This uses pre-configured AI models for image recognition, facial detection, and more.
   - **Settings**: Adjust application settings like themes and system preferences.

3. **AI Features**
   - **GPT Integration**: Use GPT for text summarization and generation.
   - **DALL·E Integration**: Generate images using DALL·E by providing a text prompt.

## Database Management

The application uses SQLite to manage the images, prompts, and gallery metadata. The following tables are used:

- **Images Table**: Stores image metadata, captions, image hashes, and other properties.
- **Prompts Table**: Stores predefined AI prompts for future use.
- **Gallery Table**: Stores information about images in the gallery.

## Security

- **LAN Security**: The application has built-in LAN-based security to ensure the app is only accessible from trusted local networks.
- **User Authentication**: (Future) Plan to add user authentication and access control.

## Current Status

- Currently working on fixing bugs and improving the system for smoother performance.
- Ongoing development for enhanced AI features like GPT and DALL·E integration.
- Future updates will include further security improvements, especially focusing on local network access.

## Development

This project is actively developed, and contributors are welcome. If you would like to contribute, feel free to fork the repo and submit a pull request.

### Contributing

1. Fork the repo
2. Create a new branch (`git checkout -b feature/feature-name`)
3. Commit your changes (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature/feature-name`)
5. Create a new Pull Request

## App Structure

The current application structure is as follows:
ImageAnalyzer/ 
├── main.py # Main application code 
├── requirements.txt # Python dependencies 
├── assets/ # Image files, icons, etc. 
│ └── logo.png 
│  ── splash_image.jpg  
├── db/ # SQLite database files 
│ └── image_analyzer.db 
├── src/ # Source code directory
├── gallery.py # Gallery management code
├── analysis.py # Image analysis code 
│ └── ai.py # AI model handling (GPT, DALL·E, etc.)
├── README.md # Project readme 
└── LICENSE # Project license

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For more information or any issues, feel free to contact the project maintainers or open an issue on GitHub.

---

Let me know if you'd like any other modifications or details added to the README!
