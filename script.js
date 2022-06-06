const CANVAS_SIZE = 256;

// get canvas
const canvas0 = document.getElementById("canvas-0");
const canvas1 = document.getElementById("canvas-1");
const canvas2 = document.getElementById("canvas-2");
const canvas3 = document.getElementById("canvas-3");

// get processed canvas
const canvas0_processed = document.getElementById("canvas-0-processed");
const canvas1_processed = document.getElementById("canvas-1-processed");
const canvas2_processed = document.getElementById("canvas-2-processed");
const canvas3_processed = document.getElementById("canvas-3-processed");

// get canvas context
const ctx0 = canvas0.getContext("2d");
const ctx1 = canvas1.getContext("2d");
const ctx2 = canvas2.getContext("2d");
const ctx3 = canvas3.getContext("2d");
const contexts = [ctx0, ctx1, ctx2, ctx3];

// get processed canvas contexts
const ctx0_processed = canvas0_processed.getContext("2d");
const ctx1_processed = canvas1_processed.getContext("2d");
const ctx2_processed = canvas2_processed.getContext("2d");
const ctx3_processed = canvas3_processed.getContext("2d");
const processed_contexts = [ctx0_processed, ctx1_processed, ctx2_processed, ctx3_processed];

const tests = ['Complex', 'Polygon', 'Clock', 'Memory']

// Add 'Loading...' to the canvas.
for (let i = 0; i < contexts.length; i++) {
    contexts[i].font = "28px sans-serif";
    contexts[i].textAlign = "center";
    contexts[i].fillText("Loading...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
}

fetch('GVT999.txt', {mode: 'no-cors'})
    .then(response => response.text())
    .then(text => JSON.parse(text))
    .then(data => {
        const images = [
            new Image(),
            new Image(),
            new Image(),
            new Image()
        ];

        // define processed images
        const processed_images = [
            new Image(),
            new Image(),
            new Image(),
            new Image()
        ];

        for (let i = 0; i < tests.length; i++) {
            images[i].onload = function() {
                imageWidth = images[i].naturalWidth;
                imageHeight = images[i].naturalHeight;

                const scale = Math.min(CANVAS_SIZE/imageWidth, CANVAS_SIZE/imageHeight);

                const scaledWidth = imageWidth * scale;
                const scaledHeight = imageHeight * scale;
                
                contexts[i].clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
                contexts[i].drawImage(images[i], (CANVAS_SIZE - scaledWidth)/2, (CANVAS_SIZE - scaledHeight)/2, scaledWidth, scaledHeight);
            };
            images[i].src = `data:image/png;base64,${data[tests[i].toLowerCase()].attempts.slice(-1)[0].image}`;

            processed_images[i].onload = function() {
                imageWidth = processed_images[i].naturalWidth;
                imageHeight = processed_images[i].naturalHeight;

                const scale = Math.min(CANVAS_SIZE/imageWidth, CANVAS_SIZE/imageHeight);

                const scaledWidth = imageWidth * scale;
                const scaledHeight = imageHeight * scale;

                processed_contexts[i].clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
                processed_contexts[i].drawImage(processed_images[i], (CANVAS_SIZE - scaledWidth)/2, (CANVAS_SIZE - scaledHeight)/2, scaledWidth, scaledHeight);
            };
            processed_images[i].src = `data:image/png;base64,${data[tests[i].toLowerCase()].attempts.slice(-1)[0].image}`;

            // fetch(`http://127.0.0.1:5000/predict/${tests[i]}`, {
            //     method: "POST",
            //     body: JSON.stringify({ image: data[tests[i].toLowerCase()].attempts.slice(-1)[0].image })
            // }).then(res => {
            //     return res.json()
            // }).then(data => {
            //     console.log(data);
            //     document.getElementById(`prediction-${i}-number`).innerHTML = data.result;
            // }).catch(err => {
            //     console.log("Request failed!", err);
            // });
        }

        const imgData = processed_contexts[0].getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);

        predict(imgData);
    })

async function predict(imgData) {
    const session = await ort.InferenceSession.create('./onnx_model.onnx');
    
    const input = new ort.Tensor('float32', new Float32Array(imgData.data), [CANVAS_SIZE, CANVAS_SIZE, 4]);
    
    const feeds = { 0: input };
    
    const results = await session.run(feeds);
    console.log(results)

    const resultsArr = results['76'].data

    const maxPrediction = resultsArr.indexOf(Math.max(...resultsArr));
    console.log(maxPrediction)

    document.getElementById(`prediction-0-number`).innerHTML = maxPrediction;
    // document.getElementById(`prediction-${i}-number`).innerHTML = data.result;

}
