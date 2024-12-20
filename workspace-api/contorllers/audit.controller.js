import AuditLog from "../models/audit.model.js";

export const createAuditLog = async (req, res) => {
    try {
        const { action, model, modelId, performedBy, details } = req.body;

        const newLog = new AuditLog({
            action,
            model,
            modelId,
            performedBy,
            details,
        });

        await newLog.save();

        res.status(201).json({
            success: true,
            message: "Audit log created successfully",
            log: newLog,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to create audit log",
            error: error.message,
        });
    }
};

export const getAuditLogs = async (req, res) => {
    try {
        const logs = await AuditLog.find().sort({ timestamp: -1 });

        res.status(200).json({
            success: true,
            logs,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch audit logs",
            error: error.message,
        });
    }
};

export const getAuditLogsByModel = async (req, res) => {
    try {
        const { model } = req.params;

        const logs = await AuditLog.find({ model }).sort({ timestamp: -1 });

        res.status(200).json({
            success: true,
            logs,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch audit logs",
            error: error.message,
        });
    }
};

export const getAuditLogsByUser = async (req, res) => {
    try {
        const { userId } = req.params;

        const logs = await AuditLog.find({ performedBy: userId }).sort({ timestamp: -1 });

        res.status(200).json({
            success: true,
            logs,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch audit logs for this user",
            error: error.message,
        });
    }
};