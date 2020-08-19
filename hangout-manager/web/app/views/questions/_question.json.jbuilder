json.extract! question, :id, :q_one, :q_two, :q_three, :created_at, :updated_at
json.url question_url(question, format: :json)
