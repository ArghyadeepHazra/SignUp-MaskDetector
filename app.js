const express = require('express');
const bodyParser = require('body-parser');
const request = require('request');
const https = require('https');
const app = express();
const port = process.env.PORT || 5500;
require('dotenv').config();

const listId = process.env.LIST_ID;
const apiKey = process.env.API_KEY;
const server = process.env.SERVER_ID;

app.use("/public", express.static(__dirname + "/public"))
app.use(express.static("public"));
app.use(bodyParser.urlencoded({ extended: true }));

app.get('/', function (req, res) {
    res.sendFile(__dirname + '/index.html');
});

// app.post("/", function (req, res) {
//     const firstName = req.body.fname;
//     const lastName = req.body.lname;
//     const email = req.body.email;
//
//     const data = {
//         members: [
//             {
//                 email_address: email,
//                 status: "subscribed",
//                 merge_fields: {
//                     FNAME: firstName,
//                     LNAME: lastName
//                 }
//             }
//         ],
//         update_existing: true,
//     };
//     const jsonData = JSON.stringify(data);
//
//     const options = {
//         method: "POST",
//         auth: `pratham:${apiKey}`,
//     };
//
//     const url = `https://${server}.api.mailchimp.com/3.0/lists/${listId}`;
//     const request = https.request(url, options, function (response) {
//         if (response.statusCode === 200) {
//             res.sendFile(__dirname + '/index2.html');
//         } else {
//             res.sendFile(__dirname + '/index2.html');
//         }
//
//         response.on("data", function (data) { });
//     });
//
//     request.write(jsonData);
//     request.end();
// });
app.post("/", function (req, res) {
    const firstName = req.body.fname;
    const lastName = req.body.lname;
    const email = req.body.email;

    const data = {
        members: [
            {
                email_address: email,
                status: "subscribed",
                merge_fields: {
                    FNAME: firstName,
                    LNAME: lastName
                }
            }
        ],
        update_existing: true,
    };
    const jsonData = JSON.stringify(data);

    const options = {
        method: "POST",
        auth: `pratham:${apiKey}`,
    };

    const url = `https://${server}.api.mailchimp.com/3.0/lists/${listId}`;
    const request = https.request(url, options, function (response) {
        if (response.statusCode === 200) {
            // Redirect to Django server
            res.redirect("http://127.0.0.1:8000");  // Replace with your Django server's URL
        } else {
            // Handle error
            res.sendFile(__dirname + '/index2.html');
        }

        response.on("data", function (data) { });
    });

    request.write(jsonData);
    request.end();
});

app.post("/failure", function (req, res) {
    res.redirect("/");
});

app.listen(port, function () {
    console.log(`Example app listening on port ${port}`);
});
