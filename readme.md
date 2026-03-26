

🏥 MediHub: AI-Powered Prescription & Appointment Manager
MediHub is a full-stack web application designed to bridge the gap between physical medical records and digital health management. It allows users to transform cluttered paper prescriptions into a structured, searchable digital history while ensuring they never miss a follow-up consultation.

🚀 Key Features
AI-Powered OCR Extraction: Users can upload images of physical prescriptions. The system uses Optical Character Recognition (OCR) to automatically extract the Doctor’s Name, Medication Details, and Next Visit Date.

Automated Appointment Reminders: Integrated with Nodemailer and Node-Cron, the system automatically scans the database daily and sends personalized email reminders to users 24 hours before their scheduled doctor’s visit.

Dynamic Health Dashboard: A personalized landing page that provides real-time statistics, including the total number of active prescriptions and unique doctors consulted.

Secure User Authentication: Features a robust login and signup system using Bcrypt for password hashing and Express-Session for secure state management.

Seamless Onboarding: New users receive an automated "Welcome Email" immediately upon registration to guide them through their first upload.

🛠️ Tech Stack
Backend: Node.js, Express.js

Database: PostgreSQL (Relational data for users and prescriptions)

Frontend: EJS (Embedded JavaScript Templates), CSS3

Automation: Node-Cron (Scheduled tasks), Nodemailer (Email engine)

Storage: Multer (Local file handling/uploads)

Security: Bcrypt (Hashing), Express-Session (Authentication)

💡 The Problem It Solves
Managing long-term health often involves multiple doctors and a paper trail of prescriptions that are easily lost or forgotten. MediHub digitizes this process, providing a centralized "Source of Truth" for medical history and using proactive notifications to improve patient adherence to follow-up visits.


Developed solely by Debarghya Sengupta.
Date- March 2026.


