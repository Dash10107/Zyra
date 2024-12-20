import mongoose from "mongoose";

const documentSchema = new mongoose.Schema(
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
      uploadedBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
      },
      fileUrl: {
        type: String, // URL for the stored file
        required: true,
      },
      description: {
        type: String,
      },
      tags: [
        {
          type: String, // Keywords for easier search
        },
      ],
      createdAt: {
        type: Date,
        default: Date.now,
      },
    },
    {
      timestamps: true,
    }
  );
  
export default mongoose.model('Document', documentSchema);
  