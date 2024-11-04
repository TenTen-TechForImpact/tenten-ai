// pages/api/invokeLambda.js
import axios from 'axios';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    // Lambda URL (API Gateway의 엔드포인트)
    const lambdaUrl = 'https://your-api-gateway-endpoint.amazonaws.com/prod/invoke'; // 여기에 실제 API Gateway URL을 입력하세요

    // 클라이언트에서 받은 데이터 (S3 URL 포함)
    const { s3Url } = req.body;

    // Lambda로 요청을 전송
    const response = await axios.post(lambdaUrl, { s3_url: s3Url });

    // Lambda 응답 반환
    res.status(200).json(response.data);
  } catch (error) {
    console.error('Error invoking Lambda:', error);
    res.status(500).json({ error: 'Failed to invoke Lambda' });
  }
}
