import mongoose from "mongoose";

const taskSchema = new mongoose.Schema(
    {
      priority: {
        type: String,
        enum: ['Low', 'Medium', 'High', 'Critical'],
        default: 'Medium',
      },
      assignee: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // User who created the task
        required: true,
      },
      assignedTo: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // User responsible for completing the task
      },
      bullets: [
        {
          point: {
            type: String,
            required: true,
          },
          isComplete: {
            type: Boolean,
            default: false,
          },
        },
      ],
      workspaceId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Workspace',
        required: true,
      },
      details: {
        type: String, // Full details of the task
        required: true,
      },
      urgency: {
        type: Number, // Numeric value for AI-driven prioritization
        min: 1,
        max: 5,
        default: 3,
      },
      deadline: {
        type: Date, // Task-specific deadline
      },
      segments: [
        {
          type: String, // Additional categorization or tags
        },
      ],
      createdAt: {
        type: Date,
        default: Date.now,
      },
      updatedAt: {
        type: Date,
        default: Date.now,
      },
    },
    {
      timestamps: true, // Automatically add createdAt and updatedAt
    }
  );
  
export default mongoose.model('Task', taskSchema);
  