import { useEffect, useRef, useState } from "react";
import Script from "next/script";

export default function Home() {
  const imgRef = useRef<HTMLImageElement>(null);
  const [result, setResult] = useState(false);
  useEffect(() => {
    const loadModel = async () => {
      if (!imgRef.current || !(window as any).mobilenet) return;
      const model = await (window as any).mobilenet.load();
      const predictions = await model.classify(imgRef.current);
      console.log("Predictions:", predictions);
      setResult(true);
    };

    loadModel();
  }, []);

  return (
    <>
      {/* 외부 스크립트 로딩 */}
      <Script
        src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"
        strategy="beforeInteractive"
      />
      <Script
        src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@1.0.0"
        strategy="beforeInteractive"
      />

      {/* 이미지와 실행 코드 */}
      <h1>Image Classification</h1>
      <img ref={imgRef} src="/cat.jpg" alt="cat" width={300} />
      <div>{result && "콘솔 창에서 결과를 확인하세요."}</div>
    </>
  );
}
