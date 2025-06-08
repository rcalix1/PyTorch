let session;
const charToIdx = {};
const maxLength = 16;

window.onload = async () => {
    session = await ort.InferenceSession.create('password_strength_cnn.onnx');
    console.log('ONNX Model loaded.');

    const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?`~ ";
    let idx = 1;
    for (let char of chars) {
        charToIdx[char] = idx++;
    }
};

function encodePassword(password) {
    const encoded = [];
    for (let i = 0; i < maxLength; i++) {
        if (i < password.length) {
            encoded.push(charToIdx[password[i]] || 0);
        } else {
            encoded.push(0); 
        }
    }
    return new Int32Array(encoded);
}

async function predictStrength() {
    const password = document.getElementById('passwordInput').value;
    const resultDiv = document.getElementById('result');

    if (password.trim() === "") {
        resultDiv.style.display = "none";
        return;
    }

    const encoded = encodePassword(password);
    const tensor = new ort.Tensor('int32', encoded, [1, maxLength]);
    const feeds = { input: tensor };
    const output = await session.run(feeds);
    const prediction = output.output.data;

    const predictedClass = prediction.indexOf(Math.max(...prediction));

    let strength = '';
    if (predictedClass === 0) strength = 'Weak';
    else if (predictedClass === 1) strength = 'Average';
    else if (predictedClass === 2) strength = 'Strong';

    resultDiv.innerText = `Predicted Strength: ${strength}`;
    resultDiv.style.display = "block";
}

function togglePassword() {
    const passwordField = document.getElementById('passwordInput');
    if (passwordField.type === "password") {
        passwordField.type = "text";
    } else {
        passwordField.type = "password";
    }
}
