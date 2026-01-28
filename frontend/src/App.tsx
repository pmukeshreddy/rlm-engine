import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Sessions from './pages/Sessions'
import SessionDetail from './pages/SessionDetail'
import ExecutionDetail from './pages/ExecutionDetail'
import NewExecution from './pages/NewExecution'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/sessions" element={<Sessions />} />
        <Route path="/sessions/:sessionId" element={<SessionDetail />} />
        <Route path="/executions/:executionId" element={<ExecutionDetail />} />
        <Route path="/new" element={<NewExecution />} />
      </Routes>
    </Layout>
  )
}

export default App
