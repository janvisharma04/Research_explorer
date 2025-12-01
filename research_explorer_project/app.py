from flask import Flask, render_template, request, redirect, url_for, flash
from crew import ResearchExplorerCrew

import os

app = Flask(__name__)
app.secret_key = os.environ.get("GOOGLE_API_KEY")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        topic = request.form.get("topic", "").strip()
        extra_instructions = request.form.get("instructions", "").strip()

        if not topic:
            flash("Please enter a research topic.", "danger")
            return redirect(url_for("index"))

        crew = ResearchExplorerCrew()
        result = crew.run(topic=topic, instructions=extra_instructions)

        return render_template(
            "report.html",
            topic=topic,
            instructions=extra_instructions,
            result=result,
        )

    return render_template("index.html")


if __name__ == "__main__":
    # You can set host="0.0.0.0" and debug=False for deployment
    app.run(debug=True)
