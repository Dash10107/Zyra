import Meeting from "../models/meeting.model.js";

export const createMeeting = async (req, res) => {
    try {
        const { title, workspaceId, participants, scheduledTime, agenda, meetingLink, recordedNotes } = req.body;

        const newMeeting = new Meeting({
            title,
            workspaceId,
            participants,
            scheduledTime,
            agenda,
            meetingLink,
            recordedNotes,
        });

        await newMeeting.save();

        res.status(201).json({
            success: true,
            message: "Meeting created successfully",
            meeting: newMeeting,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to create meeting",
            error: error.message,
        });
    }
};

export const getMeetingsByWorkspace = async (req, res) => {
    try {
        const { workspaceId } = req.params;

        const meetings = await Meeting.find({ workspaceId }).sort({ scheduledTime: -1 });

        res.status(200).json({
            success: true,
            meetings,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch meetings",
            error: error.message,
        });
    }
};

export const getMeetingDetails = async (req, res) => {
    try {
        const { id } = req.params;

        const meeting = await Meeting.findById(id);

        if (!meeting) {
            return res.status(404).json({
                success: false,
                message: "Meeting not found",
            });
        }

        res.status(200).json({
            success: true,
            meeting,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch meeting details",
            error: error.message,
        });
    }
};

export const updateMeeting = async (req, res) => {
    try {
        const { id } = req.params;
        const { title, participants, scheduledTime, agenda, meetingLink, recordedNotes } = req.body;

        const updatedMeeting = await Meeting.findByIdAndUpdate(
            id,
            {
                title,
                participants,
                scheduledTime,
                agenda,
                meetingLink,
                recordedNotes,
            },
            { new: true }
        );

        if (!updatedMeeting) {
            return res.status(404).json({
                success: false,
                message: "Meeting not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Meeting updated successfully",
            meeting: updatedMeeting,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to update meeting",
            error: error.message,
        });
    }
};

export const deleteMeeting = async (req, res) => {
    try {
        const { id } = req.params;

        const meeting = await Meeting.findByIdAndDelete(id);

        if (!meeting) {
            return res.status(404).json({
                success: false,
                message: "Meeting not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Meeting deleted successfully",
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to delete meeting",
            error: error.message,
        });
    }
};

export const addParticipant = async (req, res) => {
    try {
        const { id } = req.params;
        const { participantId } = req.body;

        const meeting = await Meeting.findById(id);

        if (!meeting) {
            return res.status(404).json({
                success: false,
                message: "Meeting not found",
            });
        }

        if (!meeting.participants.includes(participantId)) {
            meeting.participants.push(participantId);
            await meeting.save();
        }

        res.status(200).json({
            success: true,
            message: "Participant added to meeting",
            meeting,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to add participant",
            error: error.message,
        });
    }
};

export const removeParticipant = async (req, res) => {
    try {
        const { id } = req.params;
        const { participantId } = req.body;

        const meeting = await Meeting.findById(id);

        if (!meeting) {
            return res.status(404).json({
                success: false,
                message: "Meeting not found",
            });
        }

        meeting.participants = meeting.participants.filter(
            (participant) => participant.toString() !== participantId.toString()
        );
        await meeting.save();

        res.status(200).json({
            success: true,
            message: "Participant removed from meeting",
            meeting,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to remove participant",
            error: error.message,
        });
    }
};