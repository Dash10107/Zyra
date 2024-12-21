import Workspace from "../models/workspace.model.js";

export const createWorkspace = async (req, res) => {
    try {
        const { managerId, teamMembers, details, description, title, context, deadline } = req.body;

        const workspace = await Workspace.create({
            managerId,
            teamMembers,
            details,
            description,
            title,
            context,
            deadline,
        });

        res.status(201).json({
            success: true,
            message: "Workspace created successfully",
            workspace,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to create workspace",
            error: error.message,
        });
    }
};

export const getAllWorkspaces = async (req, res) => {
    try {
        const workspaces = await Workspace.find().populate("managerId teamMembers.memberId");

        res.status(200).json({
            success: true,
            workspaces,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch workspaces",
            error: error.message,
        });
    }
};

export const getWorkspaceById = async (req, res) => {
    try {
        const { id } = req.params;

        const workspace = await Workspace.findById(id).populate("managerId teamMembers.memberId");

        if (!workspace) {
            return res.status(404).json({
                success: false,
                message: "Workspace not found",
            });
        }

        res.status(200).json({
            success: true,
            workspace,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch workspace",
            error: error.message,
        });
    }
};

export const updateWorkspace = async (req, res) => {
    try {
        const { id } = req.params;
        const updates = req.body;

        const workspace = await Workspace.findByIdAndUpdate(id, updates, { new: true }).populate(
            "managerId teamMembers.memberId"
        );

        if (!workspace) {
            return res.status(404).json({
                success: false,
                message: "Workspace not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Workspace updated successfully",
            workspace,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to update workspace",
            error: error.message,
        });
    }
};

export const deleteWorkspace = async (req, res) => {
    try {
        const { id } = req.params;

        const workspace = await Workspace.findByIdAndDelete(id);

        if (!workspace) {
            return res.status(404).json({
                success: false,
                message: "Workspace not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Workspace deleted successfully",
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to delete workspace",
            error: error.message,
        });
    }
};

export const addTeamMember = async (req, res) => {
    try {
        const { id } = req.params; // Workspace ID
        const { memberId, role, weight } = req.body;

        const workspace = await Workspace.findByIdAndUpdate(
            id,
            {
                $addToSet: { teamMembers: { memberId, role, weight } }, // Prevent duplicate entries
            },
            { new: true }
        ).populate("teamMembers.memberId");

        if (!workspace) {
            return res.status(404).json({
                success: false,
                message: "Workspace not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Team member added successfully",
            workspace,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to add team member",
            error: error.message,
        });
    }
};

export const removeTeamMember = async (req, res) => {
    try {
        const { id } = req.params; // Workspace ID
        const { memberId } = req.body;

        const workspace = await Workspace.findByIdAndUpdate(
            id,
            {
                $pull: { teamMembers: { memberId } }, // Remove member
            },
            { new: true }
        ).populate("teamMembers.memberId");

        if (!workspace) {
            return res.status(404).json({
                success: false,
                message: "Workspace not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Team member removed successfully",
            workspace,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to remove team member",
            error: error.message,
        });
    }
};
