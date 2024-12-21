import express from "express";
import {
    createCalendarEvent,
    getEventsByUser,
    getEventDetails,
    updateCalendarEvent,
    deleteCalendarEvent,
    getEventsByDateRange,
} from "../contorllers/callendar.controller.js";

import isAuthenticated from "../middlewares/isAuthenticated.js";

const router = express.Router();

// CRUD and event-specific routes
router.route("/").post(isAuthenticated, createCalendarEvent); // Create a new calendar event
router.route("/:userId").get(isAuthenticated, getEventsByUser); // Get all events for a specific user
router.route("/:id").get(isAuthenticated, getEventDetails); // Get details of a specific event
router.route("/:id").put(isAuthenticated, updateCalendarEvent); // Update calendar event
router.route("/:id").delete(isAuthenticated, deleteCalendarEvent); // Delete a specific event
router.route("/:userId/dateRange").get(isAuthenticated, getEventsByDateRange); // Get events within a specific date range

export default router;
