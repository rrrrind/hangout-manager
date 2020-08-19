class CreateRecommendUsers < ActiveRecord::Migration[6.0]
  def change
    create_table :recommend_users do |t|
      t.references :user, null: false, foreign_key: true

      t.timestamps
    end
  end
end
