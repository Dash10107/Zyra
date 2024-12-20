import express from "express";
import {
    createWorkspace,
    getAllWorkspaces,
    getWorkspaceById,
    updateWorkspace,
    deleteWorkspace,
    addTeamMember,
    removeTeamMember,
} from "../contorllers/workspace.controller.js";

import isAuthenticated from "../middlewares/isAuthenticated.js";

const router = express.Router();

// Workspace CRUD operations
router.route("/").post(isAuthenticated, createWorkspace).get(isAuthenticated, getAllWorkspaces);
router
    .route("/:id")
    .get(isAuthenticated, getWorkspaceById)
    .put(isAuthenticated, updateWorkspace)
    .delete(isAuthenticated, deleteWorkspace);

// Team member operations
router.route("/:id/team-members/add").post(isAuthenticated, addTeamMember);
router.route("/:id/team-members/remove").post(isAuthenticated, removeTeamMember);

export default router;
