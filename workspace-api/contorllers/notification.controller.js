import Notification from "../models/notification.model.js";

export const createNotification = async (req, res) => {
    try {
        const { userId, type, message, link } = req.body;

        const notification = await Notification.create({
            userId,
            type,
            message,
            link,
        });

        res.status(201).json({
            success: true,
            message: "Notification created successfully",
            notification,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to create notification",
            error: error.message,
        });
    }
};

export const getNotificationsByUser = async (req, res) => {
    try {
        const { userId } = req.params;

        const notifications = await Notification.find({ userId }).sort({ createdAt: -1 });

        res.status(200).json({
            success: true,
            notifications,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch notifications",
            error: error.message,
        });
    }
};

export const getUnreadNotifications = async (req, res) => {
    try {
        const { userId } = req.params;

        const unreadNotifications = await Notification.find({ userId, read: false }).sort({ createdAt: -1 });

        res.status(200).json({
            success: true,
            unreadNotifications,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch unread notifications",
            error: error.message,
        });
    }
};

export const markNotificationAsRead = async (req, res) => {
    try {
        const { id } = req.params;

        const notification = await Notification.findByIdAndUpdate(id, { read: true }, { new: true });

        if (!notification) {
            return res.status(404).json({
                success: false,
                message: "Notification not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Notification marked as read",
            notification,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to mark notification as read",
            error: error.message,
        });
    }
};

export const deleteNotification = async (req, res) => {
    try {
        const { id } = req.params;

        const notification = await Notification.findByIdAndDelete(id);

        if (!notification) {
            return res.status(404).json({
                success: false,
                message: "Notification not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Notification deleted successfully",
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to delete notification",
            error: error.message,
        });
    }
};

export const markAllNotificationsAsRead = async (req, res) => {
    try {
        const { userId } = req.params;

        const result = await Notification.updateMany({ userId, read: false }, { read: true });

        res.status(200).json({
            success: true,
            message: `${result.nModified} notifications marked as read`,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to mark all notifications as read",
            error: error.message,
        });
    }
};
