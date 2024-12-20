import express from "express";
import {
    createDocument,
    getDocumentsByWorkspace,
    getDocumentDetails,
    updateDocument,
    deleteDocument,
    searchDocumentsByTags,
} from "../contorllers/doc.controller.js";

import isAuthenticated from "../middlewares/isAuthenticated.js";

const router = express.Router();

// CRUD and document-specific routes
router.route("/").post(isAuthenticated, createDocument); // Upload a new document
router.route("/:workspaceId").get(isAuthenticated, getDocumentsByWorkspace); // Get all documents for a specific workspace
router.route("/:id").get(isAuthenticated, getDocumentDetails); // Get details of a specific document
router.route("/:id").put(isAuthenticated, updateDocument); // Update document information
router.route("/:id").delete(isAuthenticated, deleteDocument); // Delete a specific document
router.route("/search/:tag").get(isAuthenticated, searchDocumentsByTags); // Search documents by tag

export default router;
