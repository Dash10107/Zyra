import { z } from "zod";
import { TaskStatus } from "./types";

export const createTaskSchema = z.object({
  name: z.string().min(1, { message: "Task name is required." }),
  status: z.nativeEnum(TaskStatus, { required_error: "Status is required." }),
  workspaceId: z.string().trim().min(1, { message: "Workspace ID is required." }),
  projectId: z.string().trim().min(1, { message: "Project ID is required." }),
  dueDate: z.coerce.date().refine(date => date >= new Date(), { 
    message: "Due date cannot be in the past." 
  }),
  assigneeId: z.string().trim().min(1, { message: "Assignee is required." }),
  description: z.string().optional(),
});
