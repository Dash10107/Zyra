import express from "express";
import { assignTaskToUser, getAssignedTasks, getUserProfile, getUsersInWorkspace, getUserWorkspaces, login, logout, register, updateProfile, updateUserRole } from "../contorllers/user.controller.js";
import isAuthenticated from "../middlewares/isAuthenticated.js";
import { singleUpload } from "../middlewares/mutler.js";
 
const router = express.Router();

router.route("/register").post(singleUpload,register);
router.route("/login").post(login);
router.route("/logout").get(logout);
router.route("/profile/update").post(isAuthenticated,singleUpload,updateProfile);
router.get('/:id/tasks', getAssignedTasks);
router.get('/:id/workspaces', getUserWorkspaces);
router.get('/workspaces/:id/users', getUsersInWorkspace);
router.route("/profile").get(isAuthenticated, getUserProfile);

// role management
router.route("/:id/update-role").post(isAuthenticated, updateUserRole);
router.route("/:id/assign-task").post(isAuthenticated, assignTaskToUser);



export default router;

