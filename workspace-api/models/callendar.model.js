import mongoose from "mongoose";

const calendarEventSchema = new mongoose.Schema(
    {
      userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
      },
      type: {
        type: String,
        enum: ['Task', 'Meeting', 'Personal', 'Other'],
        required: true,
      },
      title: {
        type: String,
        required: true,
      },
      description: {
        type: String,
      },
      startTime: {
        type: Date,
        required: true,
      },
      endTime: {
        type: Date,
        required: true,
      },
      recurring: {
        type: Boolean,
        default: false,
      },
      recurrencePattern: {
        type: String, // e.g., 'DAILY', 'WEEKLY', 'MONTHLY'
        required: function () {
          return this.recurring;
        },
      },
      relatedEntityId: {
        type: mongoose.Schema.Types.ObjectId, // Links to Task or Meeting if applicable
        refPath: 'type',
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
  
export default mongoose.model('CalendarEvent', calendarEventSchema);
  