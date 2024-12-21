import mongoose from 'mongoose';

const workspaceSchema = new mongoose.Schema(
  {
    managerId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User', // Assuming a User model exists
      required: true,
    },
    teamMembers: [
      {
        memberId: {
          type: mongoose.Schema.Types.ObjectId,
          ref: 'User',
          required: true,
        },
        role: {
          type: String,
          enum: ['Member', 'Manager', 'Lead'], // Adjust as needed
          required: true,
        },
        weight: {
          type: Number, // Weight for task assignment priority
          default: 1,
        },
      },
    ],
    details: {
      type: String, // Detailed description of the workspace
      default: '',
    },
    description: {
      type: String, // Short summary
      default: '',
    },
    title: {
      type: String,
      required: true,
    },
    context: {
      type: String, // Contextual data for AI-related tasks
      default: '',
    },
    deadline: {
      type: Date, // Overall deadline for the workspace
    },
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

export default mongoose.model('Workspace', workspaceSchema);
