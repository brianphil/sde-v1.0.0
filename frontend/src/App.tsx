import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Orders from "./pages/Orders";
import Fleet from "./pages/Fleet";
import RoutesPage from "./pages/RoutesPage";
import LearningMetrics from "./pages/LearningMetrics";
import Decisions from "./pages/Decisions";

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/orders" element={<Orders />} />
        <Route path="/fleet" element={<Fleet />} />
        <Route path="/trips" element={<RoutesPage />} />
        <Route path="/decisions" element={<Decisions />} />
        <Route path="/learning" element={<LearningMetrics />} />
      </Routes>
    </Layout>
  );
}

export default App;
