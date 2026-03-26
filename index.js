const express = require('express');
const path    = require('path');
const bcrypt  = require('bcrypt');
const session = require('express-session');
const { Client } = require('pg');
const multer   = require('multer');
const axios    = require('axios');
const FormData = require('form-data');
const fs       = require('fs');
const nodemailer = require('nodemailer');
const cron = require('node-cron');
require('dotenv').config();

const app       = express();
const port      = 3000;
const SALT_ROUNDS = 10;

// ── Database ──────────────────────────────────────────────────────────────────
const client = new Client({
    user:     'postgres',
    host:     'localhost',
    database: 'Prescription',
    password: '1221',
    port:     5432,
});

client.connect()
    .then(() => console.log('✅ Connected to PostgreSQL'))
    .catch(err => { console.error('❌ DB connection failed:', err.message); process.exit(1); });



//Multer
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        fs.mkdirSync('uploads', { recursive: true });
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + require('path').extname(file.originalname));
    }
});
const upload = multer({ storage, limits: { fileSize: 10 * 1024 * 1024 } });

app.use('/uploads', require('express').static('uploads'));

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

app.use(session({
    secret:            'your_secret_key', // use a long random string in production
    resave:            false,
    saveUninitialized: false,             // FIX: was true — don't save empty sessions
    rolling:           true,
    cookie: {
        maxAge:   30 * 60 * 1000,         // FIX: was 60 seconds — now 30 minutes
        httpOnly: true,                   // ADDED: prevents JS access to cookie
        secure:   false,                  // set true if using HTTPS
    },
}));

// ── View Engine ───────────────────────────────────────────────────────────────
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// ── Auth Middleware ───────────────────────────────────────────────────────────
const isAuth = (req, res, next) => {
    if (req.session.user) return next();
    res.redirect('/');
};

// ── Routes ────────────────────────────────────────────────────────────────────

// GET /  →  Login page
app.get('/', (req, res) => {
    // FIX: redirect already-logged-in users away from login page
    if (req.session.user) return res.redirect('/home');
    
    res.render('Login.ejs');
});

// GET /signup
app.get('/signup', (req, res) => {
    if (req.session.user) return res.redirect('/home');
    res.render('signup.ejs');
});

// GET /home  (protected)
// app.get('/home', isAuth, (req, res) => {
//     res.render('index.ejs', { user: req.session.user });
// });



// GET /logout
app.get('/logout', (req, res) => {
    req.session.destroy(err => {
        if (err) console.error('Logout error:', err);
        res.redirect('/');
    });
});

// POST /  →  Login
app.post('/', async (req, res) => {
    const { email, password } = req.body;

    // FIX: validate inputs before hitting the DB
    if (!email || !password) {
        return res.status(400).send('Email and password are required.');
    }

    try {
        const result = await client.query(
            'SELECT * FROM users WHERE email = $1', [email]
        );

        if (result.rows.length === 0) {
            // FIX: same message for "no user" and "wrong password" — prevents email enumeration
            return res.status(401).send('Invalid email or password.');
        }

        const user    = result.rows[0];
        const isMatch = await bcrypt.compare(password, user.password);

        if (!isMatch) {
            return res.status(401).send('Invalid email or password.');
        }

        // Regenerate session to prevent session fixation attacks
        req.session.regenerate(err => {
            if (err) return res.status(500).send('Session error.');
            req.session.user = { id: user.id, name: user.firstname };
            res.redirect('/home');
        });

    } catch (err) {
        console.error('Login error:', err.message);
        res.status(500).send('Server error.');
    }
});

// POST /signup
app.post('/signup', async (req, res) => {
    const { firstname, lastname, email, password } = req.body;

    if (!firstname || !lastname || !email || !password) {
        return res.status(400).send('All fields are required.');
    }

    if (password.length < 8) {
        return res.status(400).send('Password must be at least 8 characters.');
    }

    try {
        const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);

        // 1. Insert the new user into the database
        await client.query(
            'INSERT INTO users (firstname, lastname, email, password) VALUES ($1, $2, $3, $4)',
            [firstname, lastname, email, hashedPassword]
        );

        // 2. Prepare the Welcome Email
        const welcomeMailOptions = {
            from: '"MediHub Support" <medihubccu@gmail.com>',
            to: email, // The email the user just signed up with
            subject: `Welcome to MediHub, ${firstname}! ✨`,
            html: `
                <div style="font-family: sans-serif; max-width: 600px; margin: auto; border: 1px solid #eee; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #4f46e5;">Account Created Successfully!</h2>
                    <p>Hi ${firstname},</p>
                    <p>Welcome to <strong>MediHub</strong>. We're excited to have you on board!</p>
                    <p>Now you can easily upload your prescriptions, track your doctors, and get automated appointment reminders.</p>
                    <a href="http://localhost:3000" style="display: inline-block; padding: 10px 20px; background-color: #4f46e5; color: white; text-decoration: none; border-radius: 5px; margin-top: 10px;">Login to your Dashboard</a>
                    <hr style="border: none; border-top: 1px solid #eee; margin-top: 20px;">
                    <p style="font-size: 0.8rem; color: #888;">If you did not create this account, please report this.</p>
                </div>
            `
        };

        // 3. Send the email (don't use await so the user is redirected immediately)
        transporter.sendMail(welcomeMailOptions, (err, info) => {
            if (err) {
                console.error("Welcome email error:", err.message);
            } else {
                console.log("✅ Welcome email sent to:", email);
            }
        });

        // 4. Redirect to login page
        res.redirect('/');

    } catch (err) {
        console.error('Signup error:', err.message);
        if (err.code === '23505') {
            return res.status(409).send('That email is already registered.');
        }
        res.status(500).send('Server error during signup.');
    }
});


app.post('/upload', isAuth, upload.single('prescription'), async (req, res) => {
    if (!req.file) return res.status(400).send('No file uploaded');

    let recordId;
    try {
        const { rows } = await client.query(
            `INSERT INTO prescriptions (filename, file_path, status, user_id)
             VALUES ($1, $2, 'processing', $3) RETURNING id`,
            [req.file.filename, req.file.path, req.session.user.id]
        );
        recordId = rows[0].id;

        const form = new FormData();
        form.append('image', fs.createReadStream(req.file.path));
        //
        console.log("Targeting OCR at:", `${process.env.OCR_URL}/ocr`);
        //

        const response = await axios.post(`${process.env.OCR_URL}/ocr`, form, {
            headers: form.getHeaders(),
            timeout: 120000 
        });

        const ocrData = response.data;

        await client.query(
            `UPDATE prescriptions 
             SET ocr_text=$1, doctor_name=$2, next_visit=$3, status='done' 
             WHERE id=$4`,
            [ocrData.full_text, ocrData.doctor_name, ocrData.next_visit, recordId]
        );

        // Replace res.redirect('/record'); with this:
    // Replace your current res.json line with this:
    res.json({ 
    success: true, 
    redirectUrl: `/confirm/${recordId}` 
    });

    } catch (err) {
        console.error("=== EXTRACTION ERROR ===", err.message);
        if (recordId) {
            await client.query(`UPDATE prescriptions SET status='failed' WHERE id=$1`, [recordId]);
        }
        res.status(500).send("Extraction failed.");
    }
});



app.get('/record', isAuth, async (req, res) => {
    const { rows } = await client.query(
        'SELECT * FROM prescriptions WHERE user_id=$1 ORDER BY id DESC',
        [req.session.user.id]
    );
    res.render('Record.ejs', { prescriptions: rows });
});




app.post('/save-to-db', isAuth, async (req, res) => {
    const { id, doctor_name, next_visit } = req.body;

    try {
        // Use 'client' (not 'db')
        await client.query(
            `UPDATE prescriptions 
             SET doctor_name = $1, next_visit = $2, status = 'done' 
             WHERE id = $3 AND user_id = $4`,
            [doctor_name, next_visit, id, req.session.user.id]
        );

        res.redirect('/record');
    } catch (err) {
        console.error("Save Error:", err.message);
        res.status(500).send("Database Error: " + err.message);
    }
});






app.get('/confirm/:id', isAuth, async (req, res) => {
    try {
        const { id } = req.params;
        const { rows } = await client.query(
            'SELECT * FROM prescriptions WHERE id = $1 AND user_id = $2',
            [id, req.session.user.id]
        );

        if (rows.length === 0) return res.status(404).send("Prescription not found");

        // We only pass the 'result' object
        res.render('confirm.ejs', { result: rows[0] }); 
        
    } catch (err) {
        console.error(err);
        res.status(500).send("Server Error");
    }
});

app.post('/delete-record/:id', isAuth, async (req, res) => {
    try {
        await client.query('DELETE FROM prescriptions WHERE id = $1 AND user_id = $2', 
        [req.params.id, req.session.user.id]);
        res.redirect('/record');
    } catch (err) {
        res.status(500).send("Could not delete record.");
    }
});

// This replaces both line 101 and line 275
app.get('/home', isAuth, async (req, res) => {
    try {
        // 1. Get the count of prescriptions for the current user
        const result = await client.query(
            'SELECT COUNT(*) FROM prescriptions WHERE user_id = $1',
            [req.session.user.id]
        );

        // Convert the SQL string result to a number
        const count = parseInt(result.rows[0].count);

        // 2. Render index.ejs with ALL the data needed
        res.render('index.ejs', { 
            user: req.session.user,      // Keeps your "Welcome, Name" functionality
            prescriptionCount: count    // Adds your "Active Prescriptions" count
        });

    } catch (err) {
        console.error("Dashboard error:", err);
        // Fallback so the page still loads even if the database fails
        res.render('index.ejs', { 
            user: req.session.user, 
            prescriptionCount: 0 
        });
    }
});


// ── Email Setup ──────────────────────────────────────────────────────────────
const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: 'medihubccu@gmail.com', 
        pass: 'cvfzblszvpwejkwm' // Using your App Password
    }
});

// ── Cron Job (Daily Reminders) ────────────────────────────────────────────────
// This runs every day at 8:00 AM
cron.schedule('0 8 * * *', async () => {
    console.log('Running daily appointment check...');
    try {
        // This finds visits happening TOMORROW (1 day notice)
        const result = await client.query(
            `SELECT p.doctor_name, p.next_visit, u.email, u.firstname 
             FROM prescriptions p 
             JOIN users u ON p.user_id = u.id 
             WHERE p.next_visit = CURRENT_DATE + INTERVAL '1 day'`
        );

        result.rows.forEach(reminder => {
            const mailOptions = {
                from: '"MediHub Support" <medihubccu@gmail.com>',
                to: reminder.email,
                subject: '🩺 Appointment Reminder: Visit for Tomorrow!',
                html: `
                    <div style="font-family: sans-serif; padding: 20px; border: 1px solid #eee; border-radius: 10px;">
                        <h2 style="color: #4f46e5;">Hello, ${reminder.firstname}!</h2>
                        <p>This is a reminder from <strong>MediHub</strong>.</p>
                        <p>You have a scheduled visit with <strong>${reminder.doctor_name}</strong> tomorrow.</p>
                        <hr style="border: none; border-top: 1px solid #eee;">
                        <p style="font-size: 0.8rem; color: #888;">Don't forget to take your prescriptions with you!</p>
                    </div>
                `
            };

            transporter.sendMail(mailOptions, (error, info) => {
                if (error) console.log("Email Error:", error);
                else console.log('Email sent to ' + reminder.email);
            });
        });
    } catch (err) {
        console.error("Cron Job Error:", err);
    }
});

// ── Test Route ────────────────────────────────────────────────────────────────
// app.get('/test-email', isAuth, (req, res) => {
//     console.log("🚀 BUTTON CLICKED: The route is working!"); // <--- ADD THIS
    
//     const mailOptions = {
//         from: 'medihubccu@gmail.com',
//         to: 'medihubccu@gmail.com',
//         subject: 'Test',
//         text: 'Testing...'
//     };

//     transporter.sendMail(mailOptions, (error, info) => {
//         if (error) {
//             console.log("❌ ERROR:", error.message);
//             return res.send("Error in console");
//         }
//         console.log("✅ SENT:", info.response);
//         res.send("Check Console!");
//     });
// });








// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(port, () => {
    console.log(`🚀 Server running at http://localhost:${port}`);
});