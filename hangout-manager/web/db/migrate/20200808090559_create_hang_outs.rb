class CreateHangOuts < ActiveRecord::Migration[6.0]
  def change
    create_table :hang_outs do |t|
      t.string :name
      t.date :date
      t.time :start_time
      t.time :end_time
      t.references :user, null: false, foreign_key: true

      t.timestamps
    end
  end
end
