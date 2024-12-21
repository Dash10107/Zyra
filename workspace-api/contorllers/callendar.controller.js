import CalendarEvent from "../models/callendar.model.js";

export const createCalendarEvent = async (req, res) => {
    try {
        const { userId, type, title, description, startTime, endTime, recurring, recurrencePattern, relatedEntityId } = req.body;

        const newEvent = new CalendarEvent({
            userId,
            type,
            title,
            description,
            startTime,
            endTime,
            recurring,
            recurrencePattern,
            relatedEntityId,
        });

        await newEvent.save();

        res.status(201).json({
            success: true,
            message: "Calendar event created successfully",
            event: newEvent,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to create calendar event",
            error: error.message,
        });
    }
};

export const getEventsByUser = async (req, res) => {
    try {
        const { userId } = req.params;

        const events = await CalendarEvent.find({ userId }).sort({ startTime: 1 });

        res.status(200).json({
            success: true,
            events,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch events",
            error: error.message,
        });
    }
};

export const getEventDetails = async (req, res) => {
    try {
        const { id } = req.params;

        const event = await CalendarEvent.findById(id);

        if (!event) {
            return res.status(404).json({
                success: false,
                message: "Event not found",
            });
        }

        res.status(200).json({
            success: true,
            event,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch event details",
            error: error.message,
        });
    }
};

export const updateCalendarEvent = async (req, res) => {
    try {
        const { id } = req.params;
        const { title, description, startTime, endTime, recurring, recurrencePattern, relatedEntityId } = req.body;

        const updatedEvent = await CalendarEvent.findByIdAndUpdate(
            id,
            {
                title,
                description,
                startTime,
                endTime,
                recurring,
                recurrencePattern,
                relatedEntityId,
            },
            { new: true }
        );

        if (!updatedEvent) {
            return res.status(404).json({
                success: false,
                message: "Event not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Event updated successfully",
            event: updatedEvent,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to update event",
            error: error.message,
        });
    }
};

export const deleteCalendarEvent = async (req, res) => {
    try {
        const { id } = req.params;

        const event = await CalendarEvent.findByIdAndDelete(id);

        if (!event) {
            return res.status(404).json({
                success: false,
                message: "Event not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Event deleted successfully",
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to delete event",
            error: error.message,
        });
    }
};

export const getEventsByDateRange = async (req, res) => {
    try {
        const { userId } = req.params;
        const { startDate, endDate } = req.query;

        const events = await CalendarEvent.find({
            userId,
            startTime: { $gte: new Date(startDate) },
            endTime: { $lte: new Date(endDate) },
        }).sort({ startTime: 1 });

        res.status(200).json({
            success: true,
            events,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch events",
            error: error.message,
        });
    }
};