import express from "express";
import {
    createAuditLog,
    getAuditLogs,
    getAuditLogsByModel,
    getAuditLogsByUser,
} from "../contorllers/audit.controller.js";

import isAuthenticated from "../middlewares/isAuthenticated.js";

const router = express.Router();

// Create an audit log (called during any CRUD operation on models)
router.route("/").post(isAuthenticated, createAuditLog); // Create a new audit log

// Get all audit logs
router.route("/").get(isAuthenticated, getAuditLogs); // Get all audit logs

// Get audit logs by model type (e.g., 'User', 'Task', 'Meeting')
router.route("/model/:model").get(isAuthenticated, getAuditLogsByModel); // Get audit logs for a specific model

// Get audit logs by user ID
router.route("/user/:userId").get(isAuthenticated, getAuditLogsByUser); // Get audit logs for a specific user

export default router;
