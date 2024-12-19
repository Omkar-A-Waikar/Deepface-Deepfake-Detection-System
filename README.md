**Deepfake Detection System**
=============================

This is a Django-based project that enables **image deepfake detection** and **audio deepfake detection** using pre-trained PyTorch models. It also includes **user authentication features** like sign-up, login, and logout.

![alt text](https://github.com/Omkar-A-Waikar/Team4-SE/blob/main/UI/images/homepg_ui.jpeg)
![alt text](https://github.com/Omkar-A-Waikar/Team4-SE/blob/main/UI/images/image_ui.jpeg)
![alt text](https://github.com/Omkar-A-Waikar/Team4-SE/blob/main/UI/images/audio_ui.jpeg)

**Features**
------------

### **Core Functionalities**

1.  **Image Deepfake Detection**:
    
    *   Users can upload an image to determine if it is real or deepfake.
        
    *   Model prediction confidence is displayed alongside the result.
        
2.  **Audio Deepfake Detection**:
    
    *   Users can upload an audio file to detect if the audio is real or fake.
        
3.  **User Authentication**:
    
    *   **Sign-up**: New users can create an account.
        
    *   **Login**: Existing users can log in and view the results.
        
    *   **Logout**: Users can securely log out.
        
4.  **UI**:
    
    *   Clean, modern, and responsive UI for all pages, including forms and result displays.
        

**Technologies Used**
---------------------

*   **Backend**: Django (Python)
    
*   **Frontend**: HTML, CSS, JS, Bootstrap
    
*   **Models**: PyTorch, Tensorflow
    
*   **Database**: SQLite (default)
    
*   **Libraries**:
    
    *   torch, torchvision for model loading and inference
        
    *   librosa for audio spectrogram generation
        
    *   matplotlib and Pillow for image handling
        
    *   Django built-in libraries for authentication and file handling
        

**Setup Instructions**
----------------------

### **1\. Prerequisites**

Make sure you have the following installed on your system:

*   Python (>= 3.8)
    
*   pip (Python package manager)
    
*   Virtualenv (optional but recommended)
    
*   SQLite (comes pre-installed with Django)
    
*   Git
    

### **2\. Clone the Repository**
`git clone https://github.com/yourusername/deepfake-detection.git  
cd UI `

### **3\. Create a Virtual Environment**

` python3 -m venv venv  source venv/bin/activate   # For Linux/Mac  venv\Scripts\activate       # For Windows   `

### **4\. Install Dependencies**

Install all required Python packages using pip:
 ` pip install -r requirements.txt `
