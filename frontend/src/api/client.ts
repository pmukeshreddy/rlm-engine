import axios from 'axios'

// In production, use the full backend URL; in development, use the proxy
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types
export interface Session {
  id: string
  name: string | null
  context_metadata: Record<string, unknown> | null
  created_at: string
  updated_at: string
  memory_count: number
}

export interface Execution {
  id: string
  session_id: string | null
  user_query: string
  context_size: number
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  started_at: string | null
  completed_at: string | null
  total_input_tokens: number
  total_output_tokens: number
  total_cost_usd: number
  final_result: string | null
  error_message: string | null
}

export interface ExecutionDetail extends Execution {
  tree: ExecutionTree | null
  generated_code: string | null
}

export interface ExecutionNode {
  id: string
  execution_id: string
  parent_node_id: string | null
  node_type: 'root' | 'child'
  depth: number
  sequence_number: number
  prompt: string | null
  generated_code: string | null
  status: string
  started_at: string | null
  completed_at: string | null
  model_used: string | null
  input_tokens: number
  output_tokens: number
  cost_usd: number
  output: string | null
  error_message: string | null
  memory_before: Record<string, unknown> | null
  memory_after: Record<string, unknown> | null
  children: ExecutionNode[]
}

export interface ExecutionTree {
  execution_id: string
  tree: ExecutionNode | null
  total_nodes: number
}

export interface UsageStats {
  total_executions: number
  total_input_tokens: number
  total_output_tokens: number
  total_cost_usd: number
  average_cost_per_execution: number
  executions_by_status: Record<string, number>
}

// API functions
export const getSessions = async () => {
  const { data } = await api.get<{ sessions: Session[]; total: number }>('/sessions')
  return data
}

export const getSession = async (id: string) => {
  const { data } = await api.get<Session>(`/sessions/${id}`)
  return data
}

export const createSession = async (params: {
  name?: string
  context?: string
  context_metadata?: Record<string, unknown>
}) => {
  const { data } = await api.post<Session>('/sessions', params)
  return data
}

export const updateSession = async (id: string, params: {
  name?: string
  context?: string
  context_metadata?: Record<string, unknown>
}) => {
  const { data } = await api.put<Session>(`/sessions/${id}`, params)
  return data
}

export const deleteSession = async (id: string) => {
  await api.delete(`/sessions/${id}`)
}

export const getSessionMemory = async (id: string) => {
  const { data } = await api.get<{ memory: Record<string, unknown> }>(`/sessions/${id}/memory`)
  return data.memory
}

export const getExecutions = async (sessionId?: string) => {
  const params = sessionId ? { session_id: sessionId } : {}
  const { data } = await api.get<{ executions: Execution[]; total: number }>('/executions', { params })
  return data
}

export const getExecution = async (id: string) => {
  const { data } = await api.get<ExecutionDetail>(`/executions/${id}`)
  return data
}

export const getExecutionTree = async (id: string) => {
  const { data } = await api.get<ExecutionTree>(`/executions/${id}/tree`)
  return data
}

export const createExecution = async (params: {
  user_query: string
  context?: string
  session_id?: string
  model?: string
}) => {
  const { data } = await api.post<Execution>('/execute', params)
  return data
}

export const getStats = async (sessionId?: string) => {
  const params = sessionId ? { session_id: sessionId } : {}
  const { data } = await api.get<UsageStats>('/stats', { params })
  return data
}

export default api
