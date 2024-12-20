import express from "express";
import cookieParser from "cookie-parser";
import cors from "cors";
import dotenv from "dotenv";
import connectDB from "./utils/db.js";
import userRoute from "./routes/user.route.js";
import workspaceRoute from "./routes/workspace.route.js";
import taskRoute from "./routes/task.route.js";
import notificationRoute from "./routes/notification.route.js";
import meetingRoute from "./routes/meeting.route.js";
import docRoute from "./routes/doc.route.js";
import callendarRoute from "./routes/callendar.route.js";
import auditRoute from "./routes/audit.route.js";

dotenv.config({});

const app = express();

// middleware
app.use(express.json());
app.use(express.urlencoded({extended:true}));
app.use(cookieParser());
const corsOptions = {
    origin:'http://localhost:5173',
    credentials:true
}

app.use(cors(corsOptions));

const PORT = process.env.PORT || 8000;


// api's
app.use("/api/v1/user", userRoute);
app.use("/api/v1/workspace", workspaceRoute);
app.use("/api/v1/task", taskRoute);
app.use("/api/v1/notification", notificationRoute);
app.use("/api/v1/meeting", meetingRoute);
app.use("/api/v1/doc", docRoute);
app.use("/api/v1/callendar", callendarRoute);
app.use("/api/v1/audit", auditRoute);



app.listen(PORT,()=>{
    connectDB();
    console.log(`Server running at port ${PORT}`);
})


