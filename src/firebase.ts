// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyBaKZ-RpyPvpX6GlhcCjIBuyp6DQ6L8hoc",
    authDomain: "learnmlfast.firebaseapp.com",
    projectId: "learnmlfast",
    storageBucket: "learnmlfast.firebasestorage.app",
    messagingSenderId: "72406775274",
    appId: "1:72406775274:web:509d439f40929387f72e93",
    measurementId: "G-T8FYRPD675"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);