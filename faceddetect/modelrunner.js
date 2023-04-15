const express = require('express');
const app = express();

app.get('/',function (req,res)){
    res.sendFile(__dir+"/index2.html");

}