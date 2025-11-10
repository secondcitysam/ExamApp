console.log("📘 Exam Page Loaded. ExamId:", window.examId);

if (!window.examId) {
  console.error("❌ Exam ID is missing — cannot fetch questions!");
} else {
  fetch(`/getExamQuestions?exam_id=${window.examId}`)
    .then(res => res.json())
    .then(data => {
      console.log("✅ Questions fetched:", data);
      window.questions = data || [];
      if (!window.questions.length) {
        document.getElementById("questionContainer").innerHTML =
          "<p class='text-center text-danger'>⚠️ No questions found for this exam.</p>";
      } else {
        showQuestion(0);
      }
    })
    .catch(err => {
      console.error("Error fetching questions:", err);
    });
}
