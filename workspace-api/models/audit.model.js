import mongoose from "mongoose";

const auditLogSchema = new mongoose.Schema(
    {
      action: {
        type: String,
        enum: ['CREATE', 'UPDATE', 'DELETE'],
        required: true,
      },
      model: {
        type: String,
        enum: ['User', 'Task', 'Workspace', 'Meeting', 'Document'],
        required: true,
      },
      modelId: {
        type: mongoose.Schema.Types.ObjectId,
        required: true,
      },
      performedBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true,
      },
      details: {
        type: String, // Additional details about the action
      },
      timestamp: {
        type: Date,
        default: Date.now,
      },
    }
  );
  
export default mongoose.model('AuditLog', auditLogSchema);
  