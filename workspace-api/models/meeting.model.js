import mongoose from "mongoose";

const meetingSchema = new mongoose.Schema(
    {
      title: {
        type: String,
        required: true,
      },
      workspaceId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Workspace',
        required: true,
      },
      participants: [
        {
          type: mongoose.Schema.Types.ObjectId,
          ref: 'User',
        },
      ],
      scheduledTime: {
        type: Date,
        required: true,
      },
      agenda: {
        type: String, // Agenda for the meeting
        required: true,
      },
      meetingLink: {
        type: String, // URL for online meetings (Zoom, Google Meet, etc.)
      },
      recordedNotes: {
        type: String, // Notes captured during the meeting
      },
      createdAt: {
        type: Date,
        default: Date.now,
      },
    },
    {
      timestamps: true,
    }
  );
  
export default mongoose.model('Meeting', meetingSchema);
  