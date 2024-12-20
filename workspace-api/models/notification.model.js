import mongoose from "mongoose";

const notificationSchema = new mongoose.Schema(
    {
      userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
      },
      type: {
        type: String,
        enum: ['Task', 'Meeting', 'Workspace'],
        required: true,
      },
      message: {
        type: String,
        required: true,
      },
      read: {
        type: Boolean,
        default: false,
      },
      link: {
        type: String, // URL to the related resource (e.g., Task or Meeting)
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
  
export default mongoose.model('Notification', notificationSchema);
  