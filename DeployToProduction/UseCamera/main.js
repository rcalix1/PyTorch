document.addEventListener('DOMContentLoaded', function () {
  const videoElement = document.getElementById('webcam-video');
  const startStopButton = document.getElementById('start-stop-btn');
  //const uploadButton = document.getElementById('upload-btn');
  //const imageContainer = document.getElementById('image-container');
  const videoContainer = document.getElementById('video-container');
  //const uploadedImage = document.getElementById('uploaded-image');
  const predictionContainer = document.getElementById('prediction-container');
  const predictionText = document.getElementById('prediction');

  let isWebcamOn = false;
  let mediaStream;

  const labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "del",
    "nothing",
    "space"
  ];

  // Event listener for the start/stop button
  startStopButton.addEventListener('click', toggleWebcam);

  // Function to start or stop the webcam
  async function toggleWebcam() 
  {
    try 
    {
      if (!isWebcamOn) 
      {
          mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
          videoElement.srcObject = mediaStream;
          isWebcamOn = true;
          startStopButton.textContent = 'Stop Webcam';
          //imageContainer.style.display = 'none';
          videoContainer.style.display = 'block';
          predictionContainer.style.display = 'block';

          // Start sending frames for prediction
          sendFramesForPrediction();
      }
      else
      {
          const tracks = mediaStream.getTracks();
          tracks.forEach(track => track.stop());
          videoElement.srcObject = null;
          isWebcamOn = false;
          startStopButton.textContent = 'Start Webcam';
          videoContainer.style.display = 'none';
          predictionContainer.style.display = 'none';
          isWebcamOn = false;

          // Stop sending frames for prediction
          clearInterval(sendFramesInterval);
      }
    }
    catch (error) 
    {
        console.error('Error accessing webcam:', error);
    }
  }

  // Event listener for the file upload button
  /*
  uploadButton.addEventListener('change', function (event) 
  {
    const file = event.target.files[0];
    if (file) 
    {
      // Handle the uploaded file, e.g., display the image
      const reader = new FileReader();
      reader.onload = function (e)
      {
          // Display the uploaded image
          uploadedImage.src = e.target.result;
          imageContainer.style.display = 'block';
          videoContainer.style.display = 'none';
          predictionContainer.style.display = 'none';

          
          //console.log('Uploaded image:', e.target.result);
          const imageElement = e.target.result

          const canvas = document.createElement('canvas');
          const context = canvas.getContext('2d');

          const targetWidth = 224;
          const targetHeight = 224;
          canvas.width = targetWidth;
          canvas.height = targetHeight;

          context.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
          
          // Get the image data from the canvas
          const imageData_transformed = context.getImageData(0, 0, canvas.width, canvas.height);
          const data = imageData_transformed.data;

          const float32Array = new Float32Array(targetWidth * targetHeight);
          
          for (let i = 0; i < data.length; i += 4) {
              // Calculate grayscale value
              const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;

              // Map grayscale value to [0, 1] and store in Float32Array
              float32Array[i / 4] = gray / 255.0;
          }
          channels = 3;
          const inputTensor = new ort.Tensor('float32', float32Array, [1, channels, targetHeight, targetWidth]);

          sendImageForPrediction(inputTensor);

      };
      reader.readAsDataURL(file);
    }
  });
  */
  
  // Function to continuously send frames for prediction
  let sendFramesInterval;
  function sendFramesForPrediction()
  {
    sendFramesInterval = setInterval(() => {

      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');

      const targetWidth = 224;
      const targetHeight = 224;
      canvas.width = targetWidth;
      canvas.height = targetHeight;

      context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      
      // Get the image data from the canvas
      const imageData_transformed = context.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData_transformed.data;
      channels = 3;
      
      // Normalize pixel values by dividing by 255
      const normalizedData = new Float32Array(channels * targetWidth * targetHeight);
      
      for (let i = 0; i < data.length; i++) {
        normalizedData[i] = data[i] / 255;
      }

      // Permute channels (change from 'RGBA' to 'ARGB')
      const permutedData = new Float32Array(channels * targetWidth * targetHeight);
      
      for (let i = 0; i < data.length; i += 4) {
        permutedData[i] = normalizedData[i + 3]; // Alpha
        permutedData[i + 1] = normalizedData[i]; // Red
        permutedData[i + 2] = normalizedData[i + 1]; // Green
        permutedData[i + 3] = normalizedData[i + 2]; // Blue
      }/*
      for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
        permutedData[j] = normalizedData[i];         // Red
        permutedData[j + 1] = normalizedData[i + 1]; // Green
        permutedData[j + 2] = normalizedData[i + 2]; // Blue
        permutedData[j + 3] = 1.0;  // Set alpha to 1.0 (fully opaque)
      }*/
      
      const inputTensor = new ort.Tensor('float32', permutedData, [1, channels, targetHeight, targetWidth]);

      // Put the modified pixel data back on the canvas
      //context.putImageData(float32Array, 0, 0);

      //const imageData = canvas.toDataURL('image/jpeg');

      // Send the frame to the server for prediction
      //sendImageForPrediction(imageData);
      sendImageForPrediction(inputTensor);

    }, 1000); 
  }

  // Function to send the image to the server for prediction
  async function  sendImageForPrediction(inputTensor) 
  {
    //console.log(inputTensor)
    let session = await ort.InferenceSession.create('asl_sign.onnx');
    let feeds = {"input1":inputTensor};
    let outputMap = await session.run(feeds);
    
    let outputData = outputMap.output1.data;
    //console.log(outputData)

    let max = outputData[0];
    let maxIndex = 0;

    for (let i = 1; i < outputData.length; i++) {
        if (outputData[i] > max) {
            max = outputData[i];
            maxIndex = i;
        }
    }
    outputLabel = labels[maxIndex];
    predictionText.textContent = outputLabel;
    predictionContainer.style.display = 'block';
  }
});