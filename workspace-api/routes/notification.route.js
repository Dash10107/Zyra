import express from "express";
import {
    createNotification,
    getNotificationsByUser,
    getUnreadNotifications,
    markNotificationAsRead,
    markAllNotificationsAsRead,
    deleteNotification,
} from "../contorllers/notification.controller.js";

import isAuthenticated from "../middlewares/isAuthenticated.js";

const router = express.Router();

// CRUD and notification-specific routes
router.route("/").post(isAuthenticated, createNotification); // Create a notification
router.route("/:userId").get(isAuthenticated, getNotificationsByUser); // Get all notifications for a user
router.route("/:userId/unread").get(isAuthenticated, getUnreadNotifications); // Get unread notifications for a user
router.route("/:id/read").put(isAuthenticated, markNotificationAsRead); // Mark a specific notification as read
router.route("/:userId/mark-all-read").put(isAuthenticated, markAllNotificationsAsRead); // Mark all as read
router.route("/:id").delete(isAuthenticated, deleteNotification); // Delete a notification

export default router;
