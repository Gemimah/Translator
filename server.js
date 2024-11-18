const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');

// Initialize express app
const app = express();

// Enable CORS for frontend communication
app.use(cors());

// Set up the file storage configuration for multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './uploads/'); // Folder to store uploaded files
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname)); // Ensure unique filename
  },
});

const upload = multer({ storage: storage });

// Create uploads directory if it doesn't exist
const fs = require('fs');
const uploadsDir = './uploads';
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// POST route to handle image or video uploads
app.post('/predict', upload.fields([{ name: 'image', maxCount: 1 }, { name: 'video', maxCount: 1 }]), (req, res) => {
  let fileType = '';
  let filePath = '';

  // Check if an image was uploaded
  if (req.files.image) {
    fileType = 'image';
    filePath = req.files.image[0].path; // Get the image path
  }
  // Check if a video was uploaded
  else if (req.files.video) {
    fileType = 'video';
    filePath = req.files.video[0].path; // Get the video path
  } else {
    return res.status(400).json({ error: 'No image or video file uploaded.' });
  }

  // For demonstration, we simulate predicting the sign based on the file
  const predictedSign = fileType === 'image' ? 'hello' : 'thank you';

  // In a real-world scenario, you'd process the image or video and use a machine learning model to predict the sign language gesture

  res.json({ word: predictedSign, filePath: filePath });
});

// Start the server
const port = 5000;
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});