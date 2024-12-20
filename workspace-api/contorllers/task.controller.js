import Task from "../models/task.model.js";

export const createTask = async (req, res) => {
    try {
        const { priority, assignee, assignedTo, bullets, workspaceId, details, urgency, deadline, segments } = req.body;

        const task = await Task.create({
            priority,
            assignee,
            assignedTo,
            bullets,
            workspaceId,
            details,
            urgency,
            deadline,
            segments,
        });

        res.status(201).json({
            success: true,
            message: "Task created successfully",
            task,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to create task",
            error: error.message,
        });
    }
};

export const getAllTasks = async (req, res) => {
    try {
        const tasks = await Task.find().populate("assignee assignedTo workspaceId");

        res.status(200).json({
            success: true,
            tasks,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch tasks",
            error: error.message,
        });
    }
};

export const getTaskById = async (req, res) => {
    try {
        const { id } = req.params;

        const task = await Task.findById(id).populate("assignee assignedTo workspaceId");

        if (!task) {
            return res.status(404).json({
                success: false,
                message: "Task not found",
            });
        }

        res.status(200).json({
            success: true,
            task,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch task",
            error: error.message,
        });
    }
};

export const updateTask = async (req, res) => {
    try {
        const { id } = req.params;
        const updates = req.body;

        const task = await Task.findByIdAndUpdate(id, updates, { new: true }).populate("assignee assignedTo workspaceId");

        if (!task) {
            return res.status(404).json({
                success: false,
                message: "Task not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Task updated successfully",
            task,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to update task",
            error: error.message,
        });
    }
};

export const deleteTask = async (req, res) => {
    try {
        const { id } = req.params;

        const task = await Task.findByIdAndDelete(id);

        if (!task) {
            return res.status(404).json({
                success: false,
                message: "Task not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Task deleted successfully",
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to delete task",
            error: error.message,
        });
    }
};

export const updateTaskBullet = async (req, res) => {
    try {
        const { id, bulletIndex } = req.params; // Task ID and bullet index
        const { isComplete } = req.body;

        const task = await Task.findById(id);

        if (!task) {
            return res.status(404).json({
                success: false,
                message: "Task not found",
            });
        }

        if (bulletIndex >= task.bullets.length) {
            return res.status(400).json({
                success: false,
                message: "Invalid bullet index",
            });
        }

        task.bullets[bulletIndex].isComplete = isComplete;
        await task.save();

        res.status(200).json({
            success: true,
            message: "Bullet updated successfully",
            task,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to update bullet",
            error: error.message,
        });
    }
};

export const getTasksByWorkspace = async (req, res) => {
    try {
        const { workspaceId } = req.params;

        const tasks = await Task.find({ workspaceId }).populate("assignee assignedTo");

        res.status(200).json({
            success: true,
            tasks,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch tasks by workspace",
            error: error.message,
        });
    }
};

export const getTasksByPriority = async (req, res) => {
    try {
        const { priority } = req.query; // Example: ?priority=High

        const tasks = await Task.find({ priority }).populate("assignee assignedTo");

        res.status(200).json({
            success: true,
            tasks,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch tasks by priority",
            error: error.message,
        });
    }
};