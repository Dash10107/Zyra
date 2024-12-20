import express from "express";
import {
    createMeeting,
    getMeetingsByWorkspace,
    getMeetingDetails,
    updateMeeting,
    deleteMeeting,
    addParticipant,
    removeParticipant,
} from "../contorllers/meeting.controller.js";

import isAuthenticated from "../middlewares/isAuthenticated.js";

const router = express.Router();

// CRUD and meeting-specific routes
router.route("/").post(isAuthenticated, createMeeting); // Create a meeting
router.route("/:workspaceId").get(isAuthenticated, getMeetingsByWorkspace); // Get meetings for a specific workspace
router.route("/:id").get(isAuthenticated, getMeetingDetails); // Get details of a specific meeting
router.route("/:id").put(isAuthenticated, updateMeeting); // Update a specific meeting
router.route("/:id").delete(isAuthenticated, deleteMeeting); // Delete a specific meeting
router.route("/:id/participants").post(isAuthenticated, addParticipant); // Add participant to a meeting
router.route("/:id/participants").delete(isAuthenticated, removeParticipant); // Remove participant from a meeting

export default router;
