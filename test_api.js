const https = require('https');

// Define the data you want to send in the request body
const data = JSON.stringify({
  s3_url: "s3://tenten-bucket/transcriptions/transcription.json"
});

// Set up the options for the HTTPS request
const options = {
  hostname: 'z4wbvjyfjf.execute-api.ap-northeast-2.amazonaws.com',
  port: 443,
  path: '/mvp/topic',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': data.length
  }
};

// Make the request
const req = https.request(options, (res) => {
  let responseData = '';

  // Collect response data
  res.on('data', (chunk) => {
    responseData += chunk;
  });

  // Log the response when it is complete
  res.on('end', () => {
    console.log('Response:', responseData);
  });
});

// Handle any errors
req.on('error', (error) => {
  console.error('Error:', error);
});

// Write data to request body
req.write(data);
req.end();

