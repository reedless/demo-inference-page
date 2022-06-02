const CANVAS_SIZE = 280;
const CANVAS_SCALE = 0.5;

const canvas0 = document.getElementById("canvas-0");
const canvas1 = document.getElementById("canvas-1");
const canvas2 = document.getElementById("canvas-2");
const canvas3 = document.getElementById("canvas-3");
const canvases = [canvas0, canvas1, canvas2, canvas3];

const ctx0 = canvas0.getContext("2d");
const ctx1 = canvas1.getContext("2d");
const ctx2 = canvas2.getContext("2d");
const ctx3 = canvas3.getContext("2d");
const contexts = [ctx0, ctx1, ctx2, ctx3];

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
        var image0 = new Image();
        var image1 = new Image();
        var image2 = new Image();
        var image3 = new Image();
        const images = [image0, image1, image2, image3];

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
    })