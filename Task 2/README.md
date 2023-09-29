# Video Analytics System (VAS)

A user-friendly web-based tool for comprehensive video content analysis.

## Overview
Welcome to the Video Analytics System (VAS), a user-friendly web application created for in-depth video content analysis. VAS is equipped to handle video content from a variety of sources, including YouTube video URLs and locally uploaded videos. This system is primarily focused on fundamental video content analysis, specifically:

- **People Counting:** It accurately counts the number of people in each frame.
- **Object Detection:** VAS detects objects within the video frames, highlights them with bounding boxes, and provides the names of the detected objects, all synchronized seamlessly with video playback.

## Installation
### Clone the Project

```bash
git clone https://link-to-project
cd name-project
```

### Install Dependencies

Before running VAS, ensure you have the necessary dependencies installed. Use the following commands:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/thompsondd/pytube.git
pip install -r requirements.txt
```

## Usage
To launch the VAS web application, execute the following command:

```bash
streamlit run web.py
```

Enjoy the convenience of VAS as it sets up a local host website, making it effortless to upload links from YouTube or your computer. After preprocessing, the video content becomes accessible on the website and is also saved in a designated location, typically at `link-to-your-project/results`.

## Demo
To experience the capabilities of VAS without any setup, check out our live demo at Demo Link.