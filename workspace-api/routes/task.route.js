import express from "express";
import {
    createTask,
    getAllTasks,
    getTaskById,
    updateTask,
    deleteTask,
    updateTaskBullet,
    getTasksByWorkspace,
    getTasksByPriority,
} from "../contorllers/task.controller.js";

import isAuthenticated from "../middlewares/isAuthenticated.js";

const router = express.Router();

// CRUD Operations
router.route("/").post(isAuthenticated, createTask).get(isAuthenticated, getAllTasks);
router
    .route("/:id")
    .get(isAuthenticated, getTaskById)
    .put(isAuthenticated, updateTask)
    .delete(isAuthenticated, deleteTask);

// Task-specific Operations
router.route("/:id/bullets/:bulletIndex").put(isAuthenticated, updateTaskBullet);
router.route("/workspace/:workspaceId").get(isAuthenticated, getTasksByWorkspace);
router.route("/priority").get(isAuthenticated, getTasksByPriority);

export default router;
